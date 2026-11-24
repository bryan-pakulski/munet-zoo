"""Training loop shared by the zoo's train scripts: on one GPU, or on a pool of nodes with a live dashboard.

Every script builds a model, an optimizer, a loss for one batch, the batches of an epoch and an evaluation, and hands
them to train(). The same script trains alone (--alone) or as a pool master that other machines join.
"""
import argparse
import json
import math
import os
import secrets
import socket
import tarfile
import time
import urllib.request

import numpy as np
import munet as mu
from munet import pool
from munet.checkpoint import save_state


def parser(description, *, data, epochs, batch, shard_rows, lr):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--data', default=data, help='dataset directory; missing files are downloaded into it')
    ap.add_argument('--epochs', type=int, default=epochs)
    ap.add_argument('--batch', type=int, default=batch, help='rows per optimizer step (per round with --local)')
    ap.add_argument('--shard-rows', type=int, default=shard_rows, help='rows per task a node receives')
    ap.add_argument('--lr', type=float, default=lr, help='peak learning rate; linear warmup, then cosine decay')
    ap.add_argument('--warmup', type=float, default=0.03, help='fraction of all steps spent warming up')
    ap.add_argument('--device', default='vulkan:0', help='where the master applies updates and evaluates')
    ap.add_argument('--alone', action='store_true', help='train on --device without a pool')
    ap.add_argument('--nodes', default='', help='devices of this machine that join the pool, e.g. vulkan:0,vulkan:1')
    ap.add_argument('--min-nodes', type=int, default=1, help='connected nodes to wait for before training starts')
    ap.add_argument('--local', action='store_true', help='local steps: nodes run the optimizer; more throughput, '
                                                          'not equal to single-GPU training')
    ap.add_argument('--bind', default='127.0.0.1', help='address the pool listens on (0.0.0.0: other machines)')
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--allow-http', action='store_true', help='plain HTTP to other machines (trusted network only)')
    ap.add_argument('--out', default='runs', help='directory for checkpoints and the metrics log')
    ap.add_argument('--seed', type=int, default=0)
    return ap


def fetch(url, path):
    """Download url to path unless it exists; returns path."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        print(f'downloading {url}', flush=True)
        with urllib.request.urlopen(url) as r, open(path + '.part', 'wb') as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(path + '.part', path)
    return path


def untar(url, archive, into, marker):
    """Download url to archive and extract it into `into`, unless `marker` (a path inside `into`) exists."""
    if not os.path.exists(os.path.join(into, marker)):
        with tarfile.open(fetch(url, archive)) as t:
            t.extractall(into, filter='data')


def seeded(make, seed):
    """Build a model with initial weights drawn from `seed`."""
    token = mu.nn._initialization_rng.set(np.random.default_rng(seed))
    try:
        return make()
    finally:
        mu.nn._initialization_rng.reset(token)


def evaluating(model, device, body):
    """Call body(fwd) with fwd the model's forward pass in eval mode, compiled for device, returning numpy arrays."""
    model.eval()
    try:
        fn = mu.compile(lambda *a: model(*a), device=device, precision='fp32')
        return body(lambda *a: fn(*a).numpy())
    finally:
        model.train()


def _schedule(optimizer, peak, warmup, total):
    def at(step):
        if step < warmup:
            optimizer.lr = peak * (step + 1) / warmup
        else:
            optimizer.lr = peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return at


def train(args, model, optimizer, loss_fn, epoch, steps_per_epoch, evaluate, *, name):
    """epoch(rng, batch) yields one epoch of batches (tuples of arrays cut on dim 0); evaluate(model) returns
    {metric: value} on held-out data. Saves runs/<name>/last.mnst every epoch and best.mnst for the best first metric."""
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    params = list(model.parameters())
    total = args.epochs * steps_per_epoch
    lr_at = _schedule(optimizer, args.lr, int(args.warmup * total), total)
    print(f'{name}: {sum(int(np.prod(p.shape)) for p in params) / 1e6:.2f}M parameters, {args.epochs} epochs of '
          f'{steps_per_epoch} steps, batch {args.batch}', flush=True)
    p = nodes = None
    if args.alone:
        @mu.compile(device=args.device, precision='fp32')
        def step_fn(*batch):
            optimizer.zero_grad()
            loss = loss_fn(*batch)
            loss.backward()
            optimizer.step()
            return loss

        def run(batches):
            for batch in batches:
                yield step_fn(*batch).item()
    else:
        token = os.environ.get('MUNET_POOL_TOKEN') or secrets.token_urlsafe(32)
        p = pool.Pool(loss_fn, params, optimizer, token=token, shard_rows=args.shard_rows, device=args.device,
                      lease_seconds=3600, local=args.local)
        host, port = p.serve(args.bind, args.port, allow_http=args.allow_http)
        reach = _address(host)
        scheme = 'http' if args.allow_http or host in ('127.0.0.1', 'localhost') else 'https'
        url = f'{scheme}://{reach}:{port}'
        print(f'dashboard: {url}/#token={token}', flush=True)
        print(f'join: MUNET_POOL_TOKEN={token} munet-pool-node {url} --device all'
              + (' --allow-http' if scheme == 'http' else ''), flush=True)
        if args.nodes:
            nodes = pool.Nodes(f'{scheme}://127.0.0.1:{port}', token, args.nodes, name=socket.gethostname(),
                               allow_http=True)
            nodes.start()
        while len(p.status()['nodes']) < args.min_nodes:
            time.sleep(1)

        def run(batches):
            return p.steps(batches, timeout=3600)
    rng, step, best = np.random.default_rng(args.seed), 0, None
    log = open(os.path.join(out, 'metrics.jsonl'), 'a')
    try:
        scores = evaluate(model)
        print(f'epoch 0 {_fmt(scores)}', flush=True)
        if p is not None:
            p.log(**{f'val_{k}': v for k, v in scores.items()})
        for e in range(1, args.epochs + 1):
            started, losses = time.time(), []

            def batches():
                nonlocal step
                for batch in epoch(rng, args.batch):
                    lr_at(step)
                    step += 1
                    yield batch
            for loss in run(batches()):
                losses.append(float(loss))
            seconds = time.time() - started
            scores = evaluate(model)
            record = {'epoch': e, 'train_loss': float(np.mean(losses)), 'seconds': round(seconds, 1),
                      'rows_per_s': round(len(losses) * args.batch / seconds, 1), 'lr': optimizer.lr, **scores}
            print(f'epoch {e} loss {record["train_loss"]:.4f} {seconds:.0f} s {record["rows_per_s"]} rows/s '
                  f'{_fmt(scores)}', flush=True)
            log.write(json.dumps(record) + '\n')
            log.flush()
            if p is not None:
                p.log(epoch=e, train_loss=record['train_loss'], **{f'val_{k}': v for k, v in scores.items()})
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': e}
            save_state(state, os.path.join(out, 'last.mnst'))
            key, value = next(iter(scores.items()))
            if best is None or (value < best if key.endswith('loss') else value > best):
                best = value
                save_state(state, os.path.join(out, 'best.mnst'))
    finally:
        log.close()
        if nodes is not None:
            nodes.stop()
            nodes.wait(60)
        if p is not None:
            p.close(wait=30)
    return scores


def _address(host):
    if host not in ('0.0.0.0', '', '::'):
        return host
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('192.0.2.1', 9))  # sends nothing: asks which local address routes off this machine
        return s.getsockname()[0]


def _fmt(scores):
    return ' '.join(f'{k} {v:.4f}' for k, v in scores.items())
