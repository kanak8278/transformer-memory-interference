We have done several experiments related to this phenomenon where LLMs remember the Early part more than the later parts in the prompt. We earlier build AB:AC kinda dataset and showed how LLMs behave on them. See LLM___Retrospective_interference.pdf.

Then we have started expanding our work to doing mechanistic interpretability for the same phenomenon, earlier we thought there is no one else doing similar works.
But later I talked to Petar:
Hi Petar,
Hope you are doing good.
https://arxiv.org/abs/2603.00270

We stumbled upon this effect during our routine experimental workflows, but the signal was too strong to ignore. We originally interpreted the results from a cognitive perspective to explain the interference patterns.(Edited)


Transformers Remember First, Forget Last: Dual-Process Interference in LLMs
arxiv.org
View Kanak’s profileKanak Raj
Kanak Raj  (He/Him)  9:49 PM
But now I see that you have a series of work in same. I am trying to do a mechanistic study to understand what might be causing it. Will go through your works, thanks for sharing.

View Kanak’s profileKanak Raj
Kanak Raj  (He/Him)  9:50 PM
Also, I would love to connect with you to discuss on how to further explore it.

Mar 9
Petar Veličković sent the following messages at 3:26 PM
View Petar’s profilePetar Veličković
Petar Veličković  (He/Him)  3:26 PM
Hey Kanak, thanks for getting in touch. 

We believe that it's the combination of the poor computational graph choice (as studied in our Glasses paper), coupled with the dispersion of the softmax at longer inputs (as studied in our ICML'25 paper: https://arxiv.org/abs/2410.01104) that causes representation collapse once input length sufficiently grows. Our proofs show this is unavoidable without changing the architecture.(Edited)

View Petar’s profilePetar Veličković
Petar Veličković  (He/Him)  3:34 PM
Other architectural choices like RoPE can also contribute at long lengths, see e.g. our ICLR'25 paper: https://arxiv.org/abs/2410.06205


Round and Round We Go! What makes Rotary Positional Encodings useful?
arxiv.org
View Petar’s profilePetar Veličković
Petar Veličković  (He/Him)  3:35 PM
Worth mentioning that a group of researchers in Chile have a really cool NeurIPS'25 paper showing that representation collapse is a consequence of Transformer continuity:

https://arxiv.org/abs/2505.10606

We recently used this result to show that perplexity is sometimes very gameable in LLMs:

https://arxiv.org/abs/2601.22950

If you want to connect and chat more pls drop me an email (petarv@google.com), I do not regularly read LinkedIn :)
----------------
Now when I started doing interpretability mechanistic_probing_v2/ we fall off the rails and were not able to get correct things. Therefore we started with this: v3/ 
Now I need you to continue my work, first always find reasoning of what we want to do, why we are doing the experiment, what is the goal of the experiment, what is the hypothesis, if true what will happen, if false then what. You should explore all the possible works from internet. See there are so many pieces here to fix and make a compelete story, like you are free to use whatever is possible here, also we need to standardize and complete things one at a time.

First you need to explore the entire repo and then discuss with the User once, that what are you goign to do in high level, where is the current code and what is going to be done. Then always choose small models first to run because we are running things in my local machine.

If there is ever aws login or credentials issue run: awslogin .env($AWS_PASSWORD).

I need you to make decisive gains and improvements in the research. I need you streamline work, keep track of the changes, commit things as frequently as possible. Keep running and continuing and improving if errors or bugs, again research more, read more and more things to improve what we want to do. 

Assume that after user has given you green light, user has went to sleep and you should keep working and not stop, never stop, you should just keep working, if everything is finished then bring new ideas, but never stop, never ask for permissions, just keep going.

 You should see that there are several versions of dataset and we need to really stay clear of the multi-token, single token, narrative etc, etc. Also if there is any           
  confusion in some older numbers, you should run quick experiments, never long quick experiments and see the numbers. Also you have access to all the bedrock models and other   
  models from anthropic, gemini and gpt. scripts/ So if you need to run non-mechanisitic experiments like RI, PI numbers then you should run through them if there is equivalent  
  model available, as they will be quick vs doing the mdoel on our local. Also if possible wherever use the most optimized way to runt the model, mlx. Also we need to start  with the clear plan of what we want to achieve what is already there and what we will run, start from smallest of the models. Again you are free to update delete, re-write, create do anything you want but all you actions and logics and interpretation and output everything should be noted to a file as logs so that anyone can go through it to understand your actions. 


  ❯ Try to do most optimal use of the time, when running long experiments then always run it in backgriund so that in the meanwhile you can explore other papers, work on         
  implementations and testing if possible. Also you can use the https://nnsight.net/ library as well to run anything if transformerlens doesn't support or have any issue.


  ❯ Next thing is don't use the paper's or older runs which are out of v2 and v3 folders to quote, because they might not be exactly correct and you need to validate them 