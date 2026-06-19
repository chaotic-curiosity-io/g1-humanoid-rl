# How a Robot Learns to Walk: A Zero-Background Primer

*Before diving into the data and experiments in reports 01–03, read this once. Every term the later reports use is defined here.*

---

## What is a physics simulator?

A **physics simulator** is a computer program that models the real world — gravity, friction, joint stiffness, collisions — inside software. Think of it as a video game engine that takes the laws of physics seriously: if a robot tips too far forward, it falls, exactly as it would in a factory.

We train in simulation — specifically in **MuJoCo** (short for Multi-Joint dynamics with Contact) — rather than on a real robot for three reasons:

1. **Speed.** A simulated robot can run at many times real speed. What would take weeks of physical practice can happen in hours on a fast computer.
2. **Safety.** A simulated robot that crashes costs nothing. Dropping a real G1 costs time and money and risks damaging expensive hardware.
3. **Parallelism.** A simulator running on a graphics processing unit (GPU) — the same kind of chip that renders video-game graphics — can run *thousands* of robots simultaneously. In our setup, **2048 copies of the robot practice in parallel on a single GPU (an NVIDIA GB10)**. Each copy is its own independent experiment, all feeding experience back to the same learning algorithm at once.

---

## What is the Unitree G1?

The **Unitree G1** is a human-shaped ("humanoid") robot roughly the size of a teenager. Like a human body, it has legs, hips, a torso, and arms. At each joint — knee, ankle, hip, shoulder — there is an electric motor.

"Controlling" the robot means telling those motors what to do. Specifically, the controller sends **target joint angles** (the position each motor should try to reach) roughly **50 times per second**. That is all. There is no "walk forward" command. There is only: *here is where each joint should point right now.* Making those 50-times-a-second numbers add up to smooth walking is exactly what learning has to figure out.

---

## What is a policy?

The robot's "brain" is called a **policy**. A policy is a function: it takes in everything the robot can sense — its body tilt, joint positions, joint speeds, and the velocity it has been told to reach — and outputs the joint targets to send to the motors.

We implement the policy as a **neural network**, a mathematical structure made of thousands of simple arithmetic operations stacked in layers. The network has internal numbers called **weights**. When the weights have the right values, the network produces good joint targets; when they are random, the network produces random twitches.

At the start of training, every weight is initialized at random. The freshly created policy is useless — the robot immediately falls over.

---

## What is a reward?

Instead of writing out the rules of walking by hand (which would be enormously complex), we define a **reward**: a numerical score awarded at each timestep that captures what "doing well" means.

Our reward rewards the robot for:
- Moving at the commanded forward speed
- Keeping its body upright
- Moving its joints smoothly (penalizing — subtracting points for — jerky, high-energy motions)
- Not falling

We never write a single line of code that says "swing the left leg forward." All of that emerges from chasing a higher score.

---

## How does the robot actually learn? (PPO, hand-waved)

The algorithm that adjusts the weights is called **PPO** (Proximal Policy Optimization), a widely used method in **reinforcement learning** — the field of machine learning (teaching computers to improve at a task from experience) concerned with agents that learn by trial and error.

The loop is simple in concept:

1. **Try.** Run the current policy in the simulator. Collect what happened: every observation, every action, every reward.
2. **Score.** Add up the rewards. Which decisions led to better outcomes?
3. **Nudge.** Adjust the weights slightly so that decisions that led to higher rewards become more likely next time.
4. **Repeat.** Billions of times.

No one programs a gait. No one draws a trajectory. Walking *emerges* from the network learning that certain joint-angle sequences reliably earn more reward than others.

---

## What is an episode?

The robot does not practice forever in one go. Training is broken into **episodes** — individual attempts. An episode ends in one of two ways:

- **Time-out.** The robot stayed upright long enough that the timer ran out (~20 simulated seconds). This is success.
- **Termination (fell over).** The robot's torso hit the ground. The attempt is cut short.

**Episode length** — how many timesteps the robot survived before one of these endings — is a direct measure of progress. Early in training, episodes are very short: the robot falls almost immediately. As training proceeds, episodes grow longer, meaning the robot is staying upright more often and for longer.

---

## What can you change to explore?

Report 03 ([03-turning-the-knobs.md](03-turning-the-knobs.md)) walks through what happens when you change one thing in the reward or training setup. The most instructive knobs include: how heavily falling is penalized, the weight on the velocity-tracking reward, and the length of the episode time-out. Small changes produce surprisingly different walking styles.

---

## Map of the four reports

| Report | Title | What it covers |
|--------|-------|----------------|
| **00** | *This document* | Vocabulary: simulator, policy, reward, PPO, episode |
| **01** | [Watching it learn](01-watching-it-learn.md) | Side-by-side video of the robot at different training stages, from random twitching to walking |
| **02** | [Reproducing the benchmark](02-reproducing-the-benchmark.md) | Running the training yourself and reading the learning curves |
| **03** | [Turning the knobs](03-turning-the-knobs.md) | Changing one reward parameter and observing the effect on gait and curves |

---

*All experiments use the Unitree G1 on flat terrain, trained with the MuJoCo-Warp simulator on a DGX Spark.*
