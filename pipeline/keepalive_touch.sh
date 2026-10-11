#!/usr/bin/env bash
# Refresh timestamps across the SimFoundry working set so an age-based scratch
# purge does not reclaim it.
#
# Why this is needed at all: /scratch is NFS mounted with `relatime`, so simply
# reading a file does NOT refresh its atime (verified empirically 2026-10-03 --
# a `cat` left atime unchanged; an explicit `touch -a` moved it). So normal use
# would not keep these files looking "active".
#
# Touches BOTH atime and mtime because NEU's purge criterion is not documented
# anywhere I could find, and -a alone is useless against an mtime-based policy.
# Side effects of bumping mtime are benign: Python may regenerate .pyc once per
# run, and the next `git status` in the repo re-hashes. Nothing does incremental
# builds here.
set -uo pipefail
ROOT=/scratch/pandya.kei
LOG=$ROOT/logs/keepalive.log
mkdir -p "$ROOT/logs"

start=$(date +%s)
# -xdev: never cross into another filesystem.
# Parallel xargs: ~280k files on this NFS mount is I/O-bound, not CPU-bound.
n=$(find "$ROOT" -xdev \( -type f -o -type d \) -print0 2>/dev/null \
    | tee >(tr -dc '\0' | wc -c > /tmp/.ka_count.$$) \
    | xargs -0 -P 8 -n 500 touch -a -m 2>/dev/null; cat /tmp/.ka_count.$$)
rm -f /tmp/.ka_count.$$
end=$(date +%s)
printf '%s  touched %s paths in %ss\n' "$(date -Iseconds)" "${n:-?}" "$((end-start))" >> "$LOG"
