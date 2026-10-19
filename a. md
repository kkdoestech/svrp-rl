1. The Objective (Equation 1)

The Goal: What are we trying to minimize?
minimize∑i,jcijxij+R(x)
minimizei,j∑​cij​xij​+R(x)

Think of this as the "Total Cost Formula." Whenever a delivery truck drives around, it generates costs. We want this number to be as small as possible.

Let's break down the symbols:

    ∑i,j∑i,j​ (The Sum): This means "add up everything for all possible paths." Imagine a map with points. We are looking at every possible road between point ii and point jj.

    cijcij​ (Travel Cost): This is the cost (gas, time, distance) to drive from point ii to point jj. Crucial detail: In this paper, this is stochastic—meaning it changes randomly! It's not a fixed number like in standard OR-Tools.

    xijxij​ (The Decision): This is a binary variable.

        If the truck does drive from ii to jj, xij=1xij​=1.

        If it doesn't, xij=0xij​=0.

        So, cij×xijcij​×xij​ means "add the travel cost only if we actually use that road."

    R(x)R(x) (The Recourse Penalty): "Recourse" just means "a backup plan." In the real world, if a truck arrives at a customer and the customer suddenly wants way more goods than expected, the truck might run out. It then has to drive all the way back to the depot, reload, and drive back. R(x)R(x) is the extra penalty cost for this "oops" moment.

Summary of Equation 1:
Minimize the total driving costs + the penalty costs for running out of goods.
2. The Stochastic Model (The "Weather" Equation)

The Mystery: How do we model randomness?
θi=θˉi+∑m∑nαimnwmwn+ϵi
θi​=θˉi​+m∑​n∑​αimn​wm​wn​+ϵi​

In standard VRP (like OR-Tools), you know exactly how many boxes Customer ii needs. In SVRP, you don't. This equation explains how the paper generates the random demand and random travel times.

Let's break it down piece by piece:

    θiθi​ (Theta): This is the random thing we care about. It could be the actual demand of customer ii (e.g., they need 15 boxes), or the actual travel time from ii to somewhere else.

    θˉiθˉi​ (Theta-bar): This is the Base Value. If the weather is completely neutral, this is the average amount Customer ii usually wants.

    ∑m∑nαimnwmwn∑m​∑n​αimn​wm​wn​ (The Weather Impact):

        wmwm​ and wnwn​: These are the weather variables (Temperature, Pressure, Humidity). The AI agent can see these.

        wmwnwm​wn​: This is a multiplication. Why? Because weather interacts! A "hot and humid" day is different from a "hot and dry" day. Multiplying them captures this combined effect.

        αimnαimn​ (Alpha): This is a "secret weight" that tells us exactly how much the weather affects Customer ii. The AI does not know this number. It has to learn it by experience.

    ϵiϵi​ (Epsilon): This is pure Random Noise. It's the unpredictable stuff—like a sudden traffic jam, or a customer randomly deciding to order 5 extra boxes. Nobody can predict this.

Summary of this equation:
Actual Demand = (Base Demand) + (Predictable Weather Effects) + (Pure Random Chaos).

The Key Point: The AI agent doesn't know the "secret weights" (αα) or the random chaos (ϵϵ). But because it can see the weather (ww), it can figure out the pattern. For example, it learns: "Ah, when wmwm​ (temperature) goes up, ice cream demand goes up."

















