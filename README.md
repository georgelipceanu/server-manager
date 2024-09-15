# Server Manager

## Problem Overview
There is one decision-maker who is in charge of four data-centers. Each data-center can contain two types of server: CPU, and GPU servers. The decision-maker has three objectives: to maximize servers' utilization, to maximize servers' lifespan, and to maximize the profit. At the same time, the decision-maker has to comply with one constraint: each data-center has a fixed-size capacity in terms of the number of servers it can host. In order to achieve the objectives the decision-maker can take 3 actions at each discrete time-step: buy a server, hold a server as it is, or dismiss it.

## Objectives
At each time-step, the decision-maker wants to maximize the objective function defined: 

$$
O = U \times L \times P
$$

### Utilisation (U)
Servers' utilization is defined as the ratio of demand $$D_{i,g}$$ for a certain latency sensitivity (i) and server generation (g) to the capacity $$Z_{i,g}$$ deployed to satisfy such demand. The numerator uses the met demand $$min(Z^f_{i,g}, D_{i,g}$$).

$$
U = \frac{1}{|I \times G|} \times \sum_{i \in I} \sum_{g \in G} \frac{\min(Z^f_{i,g}, D_{i,g})}{Z^f_{i,g}}
$$

$$f$$ represents the failure rate that is sampled from a truncated Weibull distribution with $$f ∈ [0.05, 0.1]$$. Specifically, the capacity $$Z^f_{i,g}$$ is equal to the sum ofthe capacities of all servers of generation $$g$$ deployed across all data-centers with latency sensitivity $$i$$ adjusted by the failure rate $$f$$ as follows: 

$$
Z^f_{i,g}= (1 − f) \times Zi,g.
$$

Also, servers' utilization is averaged across the total number of latency sensitivity and server generation pairs |I × G|.

The demand at time step \( t \) is defined as:

$$
D_{i,g,t} = D_{i,g,t-1} + N
$$

### Normalized Servers' Lifespan (L):
This is the ratio of the operating time $$x_s$$ to the server life expectancy $$\hat{x_s}$$.

$$
L = \frac{1}{|S|} \times \sum_{s \in S} \frac{x_s}{\hat{x_s}}
$$

where $$S$$ is all the servers on the fleet.

### Profit (P):
Profit is defined as the difference between revenue (R) and cost (C).

$$
P = R - C
$$

The revenue is calculated as:

$$
R = \sum_{i \in I} \sum_{g \in G} \min(Z^f_{i,g}, D_{i,g}) \times p_{i,g}
$$

The cost is calculated as:

$$
C = \sum_{k \in K} \sum_{s \in S_k} \left( r_s + e_s + \alpha(x_s) \right)
$$

Where:
- $$e_s = \hat{e_s} \times h_k$$
- $$\alpha(x_s) = b_s \times \left( 1 + \frac{1.5 x_s}{\hat{x_s}} \times \log_2\left( \frac{1.5 x_s}{\hat{x_s}} \right) \right)$$

## Actions
At each time-step, the decision-maker can take 3 actions to maximize the objective function. At each time-step the decision-maker can take as many actions as needed. As an example, at time-step 1 the decision-maker may choose to buy 50 CPU servers for data-center 1 and 10 GPU servers for data-center 2. 
Note: The "holding" of a server does not need to be stated within the solution as it is implied if it has not been dismissed yet.

## Demand
The timeline consists of 168 discrete time-steps. At time-step 0 data-centers are empty.

## Approach
The following solution uses a Rule-Based Heuristic approach at the moment, as it applies specific rules and conditions to manage server allocation and capacity in response to demand.

## What to work on
- Implementing constraints and/or other optimisation techniques for both Profit and Lifespan.
- Cleaning up bits of code to align more with general python syntax.
