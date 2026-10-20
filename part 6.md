Part 6: The Experiment - What they tested
    We build the AI, you've trained it, but does it actually work in the real world?
        -> we call this the Evaluation Phase.


    1. The Setup: What are we testing?
        First -> they generated fake maps with 10, 20, 50, and 100 customers. The depot is in the middle (0.5, 0.5).
            
        -> Then, to be incredibly thorough, they tested their AI under different "Settings", think of these as video game difficulty settings:
            + Inference strategy (How the AI makes decisions):
                + Greedy: The AI picks the absolute best next step right now (like a short-sighted person).
                + Sampling: The AI rolls a dice based on probabilities (gives different answers each time).
                + Beam Search: The AI looks a few moves ahead (like a chess player).
                    -> Result: Beam search won. -> Looking ahead is better.

            + Signal ratio (A/B/Γ): How much does the weather actually tell us?
                -> If the weather tells us everything, the AI should do great.
                    -> If the world is just random noise, the AI will struggle.

            + Fill rate (Φ): How full is the truck? If the truck is mostly empty (high slack), routing is easy. If the truck is packed to the brim, 1 wrong guess about demand means trip back to the depot.
            
            + A priori vs. Reoptimization:
                + A priori: Plan the whole route before you leave the depot, and stick to it, no matter what.
                + Reoptimization: Adapt on the fly. If you find out a customer wants way more boxes, change the route dynamically.
                    -> Result: Adapting is better (just like using Google Maps rerouting you around a traffic jam)
            
            + Variable estimates: How does the AI guess future demand?
                + Constant: "Customer A always orders 10 boxes".
                + k-NN


                
