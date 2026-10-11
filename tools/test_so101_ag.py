"""Self-test for SO101's calibrated assisted-grasp release threshold.

Run (no GPU, no simulator -- but it DOES need the simfoundry env for torch and
the omnigibson import):

    conda_envs/simfoundry/bin/python tools/test_so101_ag.py     # -> SO101_AG_OK

Exercise SO101._handle_assisted_grasping's substitution with stubs.

Verifies, WITHOUT a simulator, that for each gripper command the value handed
to ManipulationRobot's own limit test produces the intended applying_grasp, and
that controller.control is restored afterwards.
"""
import numpy as np, torch as th
import omnigibson.robots.so101 as m
from omnigibson.controllers import ControlType
from omnigibson.robots.manipulation_robot import ManipulationRobot

LO, HI = m.SO101_JAW_NORM_RANGE
seen = []

class Ctrl:
    control_type = ControlType.POSITION
    dof_idx = np.array([5])
    _inverted = np.array([False])
    def __init__(self, q): self._control = np.array([q], np.float32)
    @property
    def control(self): return self._control

def fake_super(self):
    # Replicate EXACTLY the lines from ManipulationRobot._handle_assisted_grasping
    c = self._controllers["gripper_0"]
    ctl = th.as_tensor(c.control)
    upper = self.joint_upper_limits[c.dof_idx]
    lower = self.joint_lower_limits[c.dof_idx]
    inv = th.as_tensor(c._inverted)
    per = th.where(inv, ctl > lower, ctl < upper)
    seen.append(bool(th.any(per)))

ManipulationRobot._handle_assisted_grasping = fake_super

class Stub(m.SO101):
    default_arm = "0"
    def __init__(self): pass
r = Stub()
r.arm_names = ["0"]
r.joint_lower_limits = th.tensor([0,0,0,0,0,LO], dtype=th.float32)
r.joint_upper_limits = th.tensor([0,0,0,0,0,HI], dtype=th.float32)

print(f"threshold = {m.SO101_AG_CLOSE_NORM}/100 = {m.so101_jaw_norm_to_rad(m.SO101_AG_CLOSE_NORM):+.4f} rad\n")
print(f"{'cmd':>6} {'jaw rad':>9} {'applying_grasp':>15}  {'expect':>7}  restored")
bad = 0
# (command in LeRobot units, what it MEANS in the training data, expected grasp)
cases = [(1.5,"home shut",True),(2.9,"home shut",True),(7.0,"closing",True),
         (11.9,"HOLD cube",True),(15.0,"HOLD cube",True),(17.5,"HOLD cube",True),
         (19.9,"closing",True),(20.1,"opening",False),(23.2,"open",False),
         (24.2,"open",False),(29.4,"RELEASE",False),(32.5,"open",False),
         (36.1,"RELEASE",False),(37.3,"max open",False)]
for g, meaning, want in cases:
    q = m.so101_jaw_norm_to_rad(g)
    c = Ctrl(q); r._controllers = {"gripper_0": c}
    seen.clear()
    m.SO101._handle_assisted_grasping(r)
    got = seen[0]
    ok_restore = float(c.control[0]) == np.float32(q)
    flag = "" if (got == want and ok_restore) else "   <-- WRONG"
    if flag: bad += 1
    print(f"{g:6.1f} {q:+9.4f} {str(got):>15}  {str(want):>7}  {ok_restore}  {meaning}{flag}")

# And the escape hatch: norm=100 must fall back to OmniGibson's stock rule.
r2 = Stub()
r2.arm_names = ["0"]
r2.joint_lower_limits = r.joint_lower_limits; r2.joint_upper_limits = r.joint_upper_limits
r2._ag_close_threshold_rad = None
for g in (1.5, 37.3):
    c = Ctrl(m.so101_jaw_norm_to_rad(g)); r2._controllers = {"gripper_0": c}
    seen.clear(); m.SO101._handle_assisted_grasping(r2)
    print(f"stock rule (threshold=None): cmd {g:5.1f} -> applying_grasp={seen[0]} (expect True)")
    if not seen[0]: bad += 1

print("\nSO101_AG_OK" if bad == 0 else f"\n{bad} FAILURES")
