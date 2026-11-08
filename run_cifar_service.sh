#!/bin/sh
# CIFAR-10 full run, supervised. The training script checkpoints after
# every epoch and resumes from the checkpoint, so this loop can restart
# it after any interruption (VM replacement kills plain background jobs)
# and only ever loses part of one epoch.
cd "$(dirname "$0")" || exit 1
while [ ! -f results/cifar10_results.json ]; do
  .venv/bin/python -u -m src.train_cifar10 --epochs 30 >> results/cifar10_full_run.log 2>&1
  [ -f results/cifar10_results.json ] && break
  echo "$(date): training exited before finishing, resuming in 15s" >> results/cifar10_full_run.log
  sleep 15
done
echo "$(date): CIFAR-10 full run complete" >> results/cifar10_full_run.log
