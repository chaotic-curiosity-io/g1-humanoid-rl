# Chapter 01 — The Big Picture

*Welcome. This is the first chapter of a series that starts from nothing and ends with a humanoid robot teaching itself to do a cartwheel. You do not need a robotics or programming background. Every term will be defined the first time it appears.*

---

## What is this project about?

A human-shaped robot called the Unitree G1 — roughly the size and build of a teenager, with joints at the knees, hips, shoulders, and ankles — taught itself to walk by practicing inside a computer. Nobody wrote rules for how it should move. Nobody choreographed the steps. The robot started with no knowledge at all, fell over immediately, and by the end of roughly three hours of simulated practice it was walking smoothly at a commanded speed.

That sentence has a lot packed into it. This chapter unpacks three of the big ideas:

1. What it means for a robot to *learn* to move, rather than being *programmed* to move.
2. What it means to practice "inside a computer" — what a simulator is.
3. Why we ran thousands of copies of the robot at the same time — what parallel environments are, and why they matter.

Later chapters will build the precise vocabulary on top of these three foundations. For now, aim for the intuition.

---

## Programmed versus learning: the difference that matters

Imagine you want a robot to walk forward.

The traditional approach is to program it by hand. An engineer opens a code file and writes, in meticulous detail, exactly what each joint should do at each moment: *at time 0.0 s, the left knee bends to 15°; at time 0.1 s, the right hip swings forward to 8°; …* — and on, for every joint, every timestep, every step of the gait. This can produce reliable, predictable motion, but it requires enormous human effort, and it breaks down the moment conditions change. A programmed gait tuned for flat concrete may fail the instant the robot steps on gravel, because nobody anticipated that case.

The learning approach is different. Instead of writing the rules, you write a *goal*: "I will give you a score at every moment, and a higher score means you are doing better." Then you set the robot loose and let it figure out the rules for itself, by trial and error.

> **Insight: the distinction is who does the engineering.**
> In a programmed robot, an engineer figures out the joint angles and hard-codes them. In a learning robot, the engineer designs the goal (the score), and the robot figures out the joint angles on its own by practicing. The engineering has not gone away — it has moved upstream, from specifying motion to specifying what "good motion" means.

The G1 in this project is firmly in the second camp. No one wrote a walking gait. The robot discovered one by practicing millions of times and gradually getting better at the score.

---

## The physics simulator: a robot you can crash for free

So the robot practices by trial and error. But practicing on a *real* robot is expensive and slow.

A real G1 costs tens of thousands of dollars. Every time it falls, it risks damaging motors, joints, or sensors. It can only practice in real time — one second of practice takes one second of clock time. And you can only run one robot at once per machine.

This is where a **simulator** enters the picture.

A simulator is a computer program that models the laws of physics — gravity, friction, joint stiffness, the way surfaces collide — inside software. Think of it as a video game engine that takes physics seriously: if a simulated robot tips too far forward, it falls, because the simulation applies the same gravitational torque a real robot would experience.

We use a simulator called **MuJoCo** (short for Multi-Joint dynamics with Contact), which is widely used in robotics research. Inside MuJoCo, a digital copy of the G1 can practice indefinitely. It falls: no damage, no cost, no repair time. It can run faster than real time — a few seconds of computation produce minutes of simulated experience. And crucially, the computer can run many copies of the robot simultaneously.

> **Insight: the simulator is the training ground, not the destination.**
> The goal is always to produce a learned brain that works on a *real* robot. The simulator is used because real-robot practice is too slow and costly for trial-and-error learning. This creates a gap — if the simulation is not realistic enough, the learned behavior may fail to transfer. For now, accept that the G1's physics model is realistic enough to produce genuine walking. The question of how well it transfers to hardware is one the field wrestles with actively, and a thread that runs through later chapters.

---

## Parallel environments: thousands of robots at once

Trial-and-error learning requires an enormous amount of experience to work. The robot has to try something, see whether it got a better score, update its strategy, and try again — and it needs to do this *hundreds of millions of times* before a coherent skill emerges. At real-time speed, on a single robot, this would take months or years.

The solution is to run many copies of the robot's simulation simultaneously — each one an independent **parallel environment**, practicing the same task at the same time.

In this project, we ran **2048 copies of the G1 in parallel on a single GPU** (a graphics processing unit — the same kind of chip that renders video game graphics, repurposed here for physics computation). Every one of those 2048 copies runs its own simulation: the robot in copy 1 might be mid-stride while the robot in copy 1000 has just fallen. All of their experiences flow into the same learning algorithm at the same time. The result is that three hours of wall-clock time produces the equivalent of thousands of hours of single-robot practice.

Picture a flight school where instead of one trainee pilot in one simulator, you have 2048 trainee pilots in 2048 simulators, all running simultaneously, all reporting what they learned back to a single instructor who synthesizes the lessons in real time. The instructor gets exponentially more feedback per hour than in any one-on-one setting.

> **Insight: parallelism is what makes GPU-era RL practical.**
> Modern GPU hardware was designed for graphics: rendering a scene involves computing the same calculation (shading a pixel) for millions of pixels at once in parallel. Simulating 2048 robot physics steps is the same kind of massively parallel arithmetic. The infrastructure built for video games accidentally became the engine for robot learning.

---

## Putting it together: the practice loop at 30,000 feet

Here is the simplest version of what happens during training:

1. **All 2048 robots start from scratch.** Their "brains" — the neural networks that decide what joint angles to use — are initialized with random numbers. The robots immediately fall over.
2. **Each robot tries something.** Whatever its random brain tells it to do, it does. Almost all of it is useless.
3. **Each robot gets a score.** At every moment, the simulation evaluates how the robot is doing and assigns a numerical score (higher is better). Staying upright scores points. Moving at the commanded speed scores points. Jerky, uncontrolled motion loses points.
4. **The learning algorithm looks at all the experience** — 2048 robots, each running for a few seconds — and adjusts the shared brain slightly. Actions that led to higher scores become a little more likely next time.
5. **Repeat.** Thousands of times. Each round, the brain improves a little. Each round, the robots stay upright a little longer and move a little more purposefully.

After enough rounds, something coherent emerges: a walking gait. Nobody programmed the gait. It fell out of the score-chasing process, emergent from trial and error at scale.

This series documents what actually happened: the curves that show learning progressing, the clips that show early stumbling giving way to smooth locomotion, the experiments where changing a single number in the score produced a completely different behavior, and the cases where the robot gamed the score in ways nobody intended — teaching itself to dive forward instead of run, because diving technically earned a higher score.

---

## What you now understand

- A **simulator** is a physics engine that models a robot's body in software, letting it practice cheaply, faster than real time, and without risk of hardware damage.
- **Parallel environments** are thousands of independent copies of the simulator running simultaneously on a GPU, multiplying the amount of experience the learning algorithm sees per hour of wall-clock time.
- The distinction between **a robot that learns and a robot that is programmed**: the learning approach moves the engineering from specifying joint angles to specifying a goal (a numerical score), and lets trial-and-error practice discover the motion.

These three ideas are the scaffold everything else rests on. The next chapter builds the vocabulary that makes the practice loop precise: what exactly is the "brain," what is the "score," what counts as an "attempt," and how does the learning algorithm decide what to adjust?

Continue to [Chapter 02 — The Vocabulary](02-the-vocabulary.md).
