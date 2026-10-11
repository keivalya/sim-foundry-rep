#!/bin/bash
# One-stop status for everything this project has running or recently finished.
#   bash /scratch/pandya.kei/simfoundry-repro/jobs/status.sh
# Reads only from Slurm + disk, so it works in any session, any terminal.
PROJ=/scratch/pandya.kei/simfoundry-repro
E=$PROJ/conda_envs/simfoundry-editor/bin/python
BC=$PROJ/results/so101/vla/bluecube

echo "================= QUEUE ================="
squeue -u "$USER" -o "%.10i %.22j %.9P %.11T %.8M %R" 2>/dev/null || echo "  (squeue unavailable)"

echo
echo "============ FINISHED (24 h) ============"
sacct -S "$(date -d '24 hours ago' +%Y-%m-%dT%H:%M)" \
      --format=JobID%12,JobName%22,State%12,Elapsed -n 2>/dev/null \
  | grep -vE "\.batch|\.extern|^ *[0-9]+\.[0-9]" | tail -15

echo
echo "========= BLUE-CUBE CAMPAIGNS ==========="
for f in "$BC"/campaign_*.txt; do
  [ -e "$f" ] || { echo "  (no campaign summary yet -- finisher has not run)"; break; }
  echo "--- $(basename "$f")"; sed 's/^/    /' "$f"
done

echo
echo "===== ALL SCORED BLUE-CUBE ROLLOUTS ====="
printf "    %-22s %9s %8s %7s %s\n" run path_mm lift_mm closed IN_BOX
for d in "$BC"/*/; do
  [ -f "$d/score.json" ] || continue
  "$E" -I -c "
import json,os
x=json.load(open('$d/score.json')); c=x.get('cube',{}); g=x.get('gripper',{}); s=x.get('success',{})
print('    %-22s %9.0f %8.0f %7d %s' % (os.path.basename('${d%/}'),
  c.get('path_length_m',0)*1000, c.get('max_lift_m',0)*1000,
  g.get('frames_closed_on_cube',0), s.get('cube_in_box_final')))" 2>/dev/null
done | sort -k3 -nr | head -15

echo
echo "============== KEY OUTPUTS =============="
for f in "$PROJ/results/videos/so101_pick_place.mp4" \
         "$PROJ/results/so101/multiview/videos/so101_grid_3x2.mp4" \
         "$PROJ/results/so101/vla/framed/videos/so101_pi05_live_pair.mp4"; do
  [ -e "$f" ] && printf "    %-62s %s\n" "${f#$PROJ/}" "$(du -h "$f"|cut -f1)"
done
ls -d "$BC"/*/videos 2>/dev/null | sed "s|^|    |" | head -5
