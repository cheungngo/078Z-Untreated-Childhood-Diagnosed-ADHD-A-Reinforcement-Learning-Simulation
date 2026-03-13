**Modest Stimulant Benefit and Larger Gains from Structural Synaptic Restoration Among Untreated Childhood-Diagnosed ADHD: A Reinforcement-Learning Simulation**

Authors:

Ngo Cheung, MBBS(HK), FHKAM(Psychiatry)

Affiliations:

¹ Independent Researcher

Corresponding Author:

Ngo Cheung, MBBS, FHKAM(Psychiatry)

Hong Kong SAR, China

Tel: 98768323

Email: info@cheungngomedical.com

**Conflict of Interest**: None declared.

**Funding Declaration**: This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

**Ethics Declaration**: Not applicable.

## Abstract

Many adults with ADHD, especially those diagnosed in childhood, do not achieve full symptom relief with first-line stimulant medication, even when treatment is adequate. This pattern suggests that differences in treatment response may reflect underlying biological variation rather than medication failure alone. Recent molecular findings indicate that ADHD may follow different developmental pathways depending on age at diagnosis, with childhood-diagnosed cases showing stronger signals related to microglial activity, immune pathways, and synaptic pruning.

To examine how these differences might shape long-term treatment outcomes, we developed a reinforcement-learning model based on an early-diagnosed ADHD profile. The model was initialized with marked synaptic sparsity and persistent internal noise, then trained on a four-choice decision task across five developmental stages of increasing cognitive demand. Four treatment conditions were tested across 20 independent runs: no treatment, stimulant treatment beginning in childhood, stimulant treatment beginning in adulthood, and adult stimulant treatment combined with a synaptic regrowth mechanism.

The untreated model reached 26.14% accuracy in adulthood. Childhood-onset stimulant treatment produced a modest improvement to 29.58%, while stimulants introduced only in adulthood resulted in little additional benefit at 26.58%. The largest improvement was observed when adult stimulants were combined with synaptic regrowth, increasing accuracy to 39.43%. This was also the only condition that reversed the model\'s extreme synaptic sparsity, whereas reductions in internal stress-related noise were similar across all stimulant-treated conditions.

These results provide a computational explanation for why some adults with childhood-diagnosed ADHD may respond only partially to standard stimulants. If early developmental pruning leaves lasting structural underconnectivity, then dopaminergic treatment alone may reduce noise without restoring the underlying network. Age at diagnosis may therefore act as a practical marker of a biologically distinct subtype of ADHD, one that may benefit more from treatments aimed at synaptic restoration than from stimulant therapy alone.

## Introduction

Attention-deficit/hyperactivity disorder (ADHD) is a common neurodevelopmental disorder, and for many people it continues beyond childhood into adult life \[1\]. Even so, the timing of first diagnosis differs widely. Some children are identified early, often when inattentive or hyperactive behaviour begins to interfere with school or family life. Others are not diagnosed until adolescence or adulthood, when growing demands on organisation, self-regulation, and work or study make these difficulties more visible \[2\]. This difference in timing has been linked not only to access to care or recognition of symptoms, but also to variation in comorbidity and day-to-day impairment. That has raised the possibility that childhood- and late-diagnosed ADHD may not be fully explained by a single, uniform biological profile.

Genetic findings have added support to that view. In a large Danish population-based cohort, Rajagopal et al \[3\] compared ADHD groups defined by age at first diagnosis and reported differences in their genetic architecture. Childhood-diagnosed cases, which included 14,878 individuals, showed greater genetic overlap with autism spectrum disorder and a higher burden of rare protein-truncating variants in constrained genes. Late-diagnosed cases, numbering 6,961, showed stronger polygenic correlations with depression and alcohol use disorder, along with a lower genetic correlation with childhood ADHD, with rg = 0.65. These findings paralleled clinical patterns, with autism comorbidity being more common in the childhood group, while internalising problems and substance-related difficulties were more evident among those diagnosed later.

More recent transcriptome-wide association studies have tried to extend these genetic results by asking what functional differences might sit behind them. Using GTEx brain expression models together with the same iPSYCH summary statistics, one analysis found that the overall transcriptomic profiles of childhood- and late-diagnosed ADHD were broadly similar, but not identical. Childhood-diagnosed ADHD showed relatively stronger signals in microglia marker genes and major histocompatibility complex pathways, which was consistent with neurodevelopmental and immune-related processes \[4\]. Late-diagnosed ADHD, in contrast, showed greater enrichment in long-term potentiation and complement cascade pathways. A second TWAS compared ADHD with autism spectrum disorder and reported that only the pairing of late-diagnosed ADHD and late-diagnosed autism showed a modest negative transcriptome-wide correlation. In that comparison, several shared risk loci, including mitochondrial and neuronal adhesion genes, showed opposite directions of effect \[5\]. This led to the proposal of a late-diagnosis anti-correlation hypothesis, in which compensatory transcriptional responses at shared loci might delay clinical recognition by partly buffering symptoms early on.

These TWAS findings add a useful functional layer to earlier genetic work, but they do not show directly how the implicated pathways shape behaviour or treatment response over time. Statistical genetics and expression-based methods are strong tools for identifying associations, yet they are less able to capture changing processes such as learning under increasing cognitive demand, the cumulative effects of pruning, or variation in response to medication across development. Computational approaches may help address that gap. Reinforcement-learning models, in particular, allow biologically informed parameters to be introduced in a controlled way, including features such as elevated synaptic pruning, ongoing neuroimmune-related noise in an early-diagnosed profile, or differences in long-term potentiation capacity, and then tested across simulated treatment conditions.

The present study used a reinforcement-learning model to represent the transcriptomic profile linked to early-diagnosed ADHD. The model began from an identical pruned network state intended to reflect stronger microglia- and major histocompatibility complex-related processes. It was then trained on a noisy four-choice decision task across five developmental phases, each with gradually increasing cognitive demands. Four treatment conditions were simulated: no treatment across development, chronic stimulant exposure beginning in childhood, stimulant treatment starting only in adulthood, and adult stimulant treatment combined with a gradient-guided synaptogenesis intervention described as ketamine-like. Outcome was measured as task accuracy, and network sparsity and internal noise were also tracked across 20 independent seeds.

The study had three aims. The first was to simulate the behavioural effects of the early-diagnosed transcriptomic profile across the lifespan. The second was to compare adult cognitive performance across the four treatment conditions. The third was to examine whether the model\'s predictions were in line with the pathway-level differences described in the TWAS studies and with established clinical patterns of stimulant response in adults with ADHD. In this way, the study aimed to connect statistical genetic findings with possible mechanisms that might help explain differences in treatment course.

## Methods

### Model implementation

The model was built in Python with PyTorch. It was designed to simulate the behavioral consequences of the early-diagnosed ADHD transcriptomic profile described in earlier work \[4\]. The purpose was to compare different treatment strategies across a simulated lifespan while keeping the main biological features from the transcriptome-wide analyses in place. Code and analysis scripts were prepared so that the simulations could be reproduced exactly.

### Network architecture

The model used a feed-forward policy network, PolicyNet, with three hidden layers containing 256, 128, and 4 units. ReLU activations were used throughout the hidden layers (Figure 1). After each hidden layer, Gaussian noise was added and scaled by a stress parameter. This noise term was used to represent the persistent neuroimmune and microglial activity associated with the childhood-diagnosed TWAS profile. The output layer produced logits for a four-choice decision task. No dropout or batch normalization was added. The architecture was intentionally kept simple so that the effects of pruning and noise could be examined without adding extra complexity from the network design.

![](media/image3.png){width="6.267716535433071in" height="3.9722222222222223in"}

***Figure 1.** Model architecture and intervention mechanisms. (A) The policy network (PolicyNet) receives a 2-dimensional state vector sampled from one of four Gaussian clusters. Two hidden layers (256 and 128 units, ReLU activation) each receive additive Gaussian noise scaled by a stress parameter σ, representing persistent neuroimmune activity. The output layer produces logits over four actions. (B) At initialisation, magnitude-based pruning removes 94% of weights (sparsity = 0.94), enforced by binary masks reapplied after every gradient step. (C) Stimulant modulation (StimMod) reduces σ by 68% and scales rewards by ×1.45. (D) Gradient-guided regrowth restores a fraction of pruned weights ranked by accumulated gradient magnitude, reinitialised with small random values (𝒩(0, 0.025²)).*

### Pruning and regrowth procedure

Pruning was handled by a separate Pruner class. This class applied magnitude-based thresholding to all weight matrices at the start of each simulation. The sparsity target was set at 0.94. This value was chosen to reflect the stronger pruning-related signals, including microglia markers and major histocompatibility complex enrichment, reported for early-diagnosed cases \[4\]. After pruning, binary masks were stored and then reapplied after every gradient update so that the structural deficit remained in place across training.

An optional regrowth step was also included. In this procedure, weights that had been set to zero were ranked according to the absolute size of their accumulated gradients over a short batch of episodes. A top fraction of these candidates, with a default of 0.58 in the ketamine-like condition, was restored. Reinstated weights were assigned small random values drawn from a normal distribution with mean 0 and standard deviation 0.025. This procedure was used to represent a synaptogenesis-like mechanism and to test whether structural recovery could improve performance.

### Task design

The model was trained on a four-class noisy bandit task adapted from Gaussian blob classification problems. On each trial, a two-dimensional state vector was sampled from one of four Gaussian clusters centered at \[-3, -3\], \[3, 3\], \[-3, 3\], and \[3, -3\]. Isotropic noise was added, with a base standard deviation of 0.8. The agent then selected one of four possible actions. A correct choice produced a reward of 1.0, and an incorrect choice produced 0.0. The task was selected because it required sustained attention and basic executive control but did not depend on working-memory load. This made it possible to focus on the effects of pruning and internal noise on policy learning.

### Learning algorithm

Policy updates were carried out with the REINFORCE algorithm \[6\], together with a simple advantage baseline. Returns were discounted with gamma set to 0.98. Within each episode, the advantage for each step was calculated as the observed return minus the mean return of that episode, divided by the standard deviation plus a small constant for numerical stability. This baseline reduced gradient variance and produced stable learning curves. Optimization used Adam with a base learning rate of 0.001. In stimulant conditions, the learning rate was multiplied by 1.8 to represent a modest long-term-potentiation-enhancing effect of medication.

### Developmental simulation

Development was modeled across five consecutive phases intended to approximate the period from childhood to adulthood. Cognitive load increased across phases by raising the noise level in the input data. The phase increments were 0.0, 0.2, 0.4, 0.6, and 0.8. Each phase included 150 episodes, and each episode contained 64 trials. Adult outcome was defined as the mean accuracy across the last 40 episodes of phase 5. This measure was used to capture asymptotic performance after developmental exposure to each treatment condition.

### Treatment conditions

![](media/image1.png){width="6.267716535433071in" height="2.8333333333333335in"}

***Figure 2.** Experimental design: four treatment arms across five developmental phases. Each arm begins from the same pruned network (sparsity = 0.94). Development is modelled across five consecutive phases (150 episodes each, 64 trials per episode), with input noise increasing by +0.2 per phase to simulate rising cognitive demand. Adult outcome is defined as mean accuracy over the final 40 episodes of Phase 5. All arms were replicated across 20 independent random seeds. Coloured segments indicate active treatment components. The dashed border in Arm 4 marks the phase in which gradient-guided regrowth is applied every 10th episode in addition to stimulant modulation.*

Four treatment arms were run, all beginning from the same initial pruned state. In the untreated condition, the stress parameter stayed fixed at 0.42 across all five phases. In the chronic-stimulant condition, the stress parameter was reduced by 68%, giving a value of 0.1344 from phase 1 onward. In this same condition, rewards were multiplied by 1.45 to model dopaminergic amplification of positive prediction errors. The late-stimulant condition followed the untreated settings during the first four phases and switched to the chronic-stimulant settings only in phase 5. The late-stimulant-plus-ketamine condition followed the late-stimulant schedule, but in addition applied gradient-guided regrowth every tenth episode during phase 5. This allowed structural recovery to be tested alongside functional modulation. Parameter choices were kept within ranges described as biologically plausible on the basis of the TWAS enrichment patterns and earlier computational psychiatry work \[7,8\].

### Replication and recorded measures

To examine robustness, each treatment arm was run across 20 independent random seeds. At the end of each run, three values were recorded: final adult accuracy, network sparsity, and current stress level. Sparsity was defined as the proportion of weights masked to zero across all layers. Stress was taken directly from the model parameter in each run.

### Statistical summary

Descriptive summaries were calculated in R using base functions. For each treatment arm, the mean, standard deviation, minimum, and maximum were reported across the 20 seeds. Ninety-five percent confidence intervals were calculated with the t-distribution using 19 degrees of freedom and a critical value of 2.093. No hypothesis tests were carried out. The aim was descriptive comparison of treatment trajectories rather than formal inference. For that reason, effect sizes were presented only as raw differences in mean accuracy.

### Reproducibility and computing environment

All simulations were run on standard CPU hardware without GPU acceleration. Random number generators for both NumPy and PyTorch were seeded explicitly at the start of each seed loop to ensure reproducibility. The full codebase, including configuration dictionaries, training loops, and analysis scripts, was prepared for release at a repository link to be inserted. The study was entirely in silico and did not involve external datasets or human participants.

## Results

### Overall pattern of simulated outcomes

All simulations began from the same initial network configuration, which was intended to reflect the early-diagnosed ADHD profile. After the five developmental phases, the untreated condition reached a mean adult accuracy of 26.14%, with a 95% confidence interval from 25.53 to 26.76. This level was close to chance performance on the four-choice task, which was 25%. Final sparsity in this arm was fixed at 0.939956 across all 20 seeds, and stress remained at 0.4200. In the context of the model, this pattern was consistent with the heavier pruning burden and the stronger microglia- and major histocompatibility complex-related signals linked to childhood-diagnosed cases \[4\].

### Comparison of treatment conditions

![](media/image2.png){width="6.267716535433071in" height="2.8194444444444446in"}

***Figure 3:** Adult Task Accuracy Across Treatment Conditions. Box plots represent the distribution of final accuracy (%) across 20 independent simulation seeds. Individual data points are overlaid to show the full distribution. The dashed line at 25% represents chance-level performance on the four-choice task. Only the Late Stimulant + Ketamine condition (combining dopaminergic modulation with gradient-guided synaptic regrowth) produced a substantial improvement over the untreated baseline.*

**Table 1. Summary Statistics -- Final Accuracy (%) \[95% CI, t-distribution, df=19\]**

| Arm                  | Mean  | Std  | Min   | Max   | CI Lower | CI Upper |
|----------------------|-------|------|-------|-------|----------|----------|
| Untreated            | 26.14 | 1.31 | 24.02 | 28.71 | 25.53    | 26.76    |
| Chronic Stimulants   | 29.58 | 1.80 | 27.58 | 35.39 | 28.74    | 30.42    |
| Late Stim Only       | 26.58 | 1.18 | 24.34 | 28.98 | 26.03    | 27.13    |
| Late Stim + Ketamine | 39.43 | 3.70 | 33.52 | 47.34 | 37.70    | 41.16    |

Early stimulant exposure produced a modest improvement over the untreated condition (Figure 3, Table 1). In the chronic-stimulant arm, mean adult accuracy increased to 29.58% (95% confidence interval: 28.74 to 30.42), which was an average gain of 3.44 percentage points. Sparsity did not change and remained at 0.939956, while stress was reduced to 0.1344.

When stimulant treatment was delayed until the final phase, performance changed very little. The late-stimulant-only arm reached a mean adult accuracy of 26.58%, with a 95% confidence interval from 26.03 to 27.13. This overlapped closely with the untreated condition. Sparsity again remained at 0.939956, and stress was reduced to 0.1344 only in the last phase.

The strongest effect was observed when adult stimulant treatment was combined with gradient-guided regrowth, the ketamine-like condition. Mean adult accuracy in this arm rose to 39.43%, with a 95% confidence interval from 37.70 to 41.16. This represented an improvement of 13.29 percentage points relative to the untreated arm. It was also the only condition in which sparsity changed substantially, dropping to 0.000089. Stress reduction was the same as in the other stimulated arms, at 0.1344. Summary values for accuracy, sparsity, and stress are presented in Tables 4, 5, and 6.

### Seed-level variation

Per-seed outcomes are shown in Tables 2. In the untreated arm, final accuracy ranged from 24.02% to 28.71% across the 20 seeds. In the chronic-stimulant arm, the range was 27.58% to 35.39%. The late-stimulant-only condition showed a similar spread, from 24.34% to 28.98%. The late-stimulant-plus-ketamine arm showed the widest range, from 33.52% to 47.34%. This wider spread appeared only after pruning had been reversed by the structural intervention.

**Table 2. Per-Seed Final Accuracy (%)**

| Seed | Untreated | Chronic Stim | Late Stim Only | Late Stim+Ket |
|------|-----------|--------------|----------------|---------------|
| 0    | 26.09     | 28.91        | 25.86          | 36.84         |
| 1    | 25.90     | 30.86        | 26.05          | 47.34         |
| 2    | 25.70     | 29.49        | 26.84          | 42.30         |
| 3    | 24.02     | 28.67        | 26.99          | 40.08         |
| 4    | 25.51     | 28.75        | 26.64          | 33.52         |
| 5    | 25.00     | 27.93        | 28.98          | 37.27         |
| 6    | 24.77     | 28.95        | 24.96          | 39.73         |
| 7    | 27.97     | 31.64        | 27.38          | 45.35         |
| 8    | 27.42     | 35.39        | 27.19          | 40.31         |
| 9    | 28.71     | 30.74        | 28.75          | 38.12         |
| 10   | 25.47     | 27.58        | 26.13          | 35.74         |
| 11   | 27.85     | 28.67        | 25.74          | 38.16         |
| 12   | 27.85     | 31.68        | 26.99          | 37.54         |
| 13   | 25.59     | 28.44        | 27.34          | 38.09         |
| 14   | 25.78     | 28.12        | 25.35          | 38.79         |
| 15   | 27.38     | 28.71        | 27.97          | 46.09         |
| 16   | 25.78     | 29.61        | 26.29          | 42.73         |
| 17   | 24.80     | 28.87        | 24.34          | 36.21         |
| 18   | 26.64     | 28.55        | 25.59          | 35.12         |
| 19   | 24.65     | 30.04        | 26.21          | 39.26         |

By contrast, sparsity and stress values were essentially identical across seeds within each arm. For untreated, chronic-stimulant, and late-stimulant-only conditions, sparsity was 0.939956 in every seed. In the late-stimulant-plus-ketamine condition, it was 0.000089 in every seed. Stress was 0.420000 in every untreated run and 0.134400 in every stimulated run. These fixed values indicate that the effects of treatment on network structure and internal noise were deterministic within the simulation.

### Functional and structural dissociation

Taken together, the findings showed a separation between functional and structural effects in the model. All stimulated conditions reduced internal noise to the same level, but only the addition of regrowth changed sparsity. In practical terms, lowering noise alone produced either a modest gain, as in the chronic-stimulant arm, or almost no rescue when treatment began late. The largest improvement appeared only when reduced stress was paired with structural restoration.

Within the framework used here, this pattern matched the distinction between dopamine-related improvements in signal-to-noise and the separate need for renewed synaptic connectivity in the early-diagnosed subtype. The limited benefit of chronic stimulant treatment, and the lack of improvement when stimulants were introduced only in adulthood, were in line with the weaker long-term potentiation enrichment and heavier pruning signals reported for childhood-diagnosed ADHD \[4\]. The clear advantage of combining stimulant exposure with structural repair therefore offers one possible explanation for why some adults diagnosed early may show only limited benefit from standard pharmacological treatment alone.

## Discussion

### Interpretation of the main findings

Three findings stood out across the simulations. Early, sustained stimulant exposure led to a small but consistent improvement in adult accuracy, with a mean increase of 3.44 percentage points over the untreated condition. Starting stimulants only in adulthood did not meaningfully alter the untreated course. The largest improvement appeared when adult stimulant treatment was combined with a single round of gradient-guided synaptogenesis, which increased accuracy by 13.29 percentage points relative to the untreated arm. These differences emerged even though sparsity stayed uniformly high in all non-ketamine conditions and stress was reduced to the same level in every stimulated condition. That pattern points to a separation between functional effects, reflected here by stress reduction, and structural effects, reflected by sparsity. This separation seems central to explaining why timing and treatment type mattered.

The modest benefit seen with chronic stimulants fits the weaker long-term potentiation signal reported for childhood-diagnosed ADHD in the transcriptomic work by Cheung \[4\]. In the present model, stimulants reduced internal noise and increased the impact of positive rewards, but they did not rebuild a network that had already lost a large number of connections. This resembles the transcriptomic contrast in which early-diagnosed cases showed only modest long-term potentiation enrichment, with an enrichment ratio of 1.10, compared with 1.34 in late-diagnosed cases \[4\]. In other words, dopaminergic modulation improved signal quality, but it did not solve the structural loss created during pruning. The failure of adult-only stimulant treatment adds to that point. Once pruning had already accumulated, functional adjustment alone was not enough to recover performance. This is broadly in line with clinical reports that adults diagnosed in childhood may need higher doses or may show smaller gains than those diagnosed later \[9\].

### Structural repair versus functional rescue

The condition that combined stimulants with ketamine-like regrowth gave the clearest indication of what the model treated as the missing ingredient. In that arm, sparsity fell from 0.939956 to 0.000089, while stress dropped by exactly the same amount as in the other stimulated conditions. The extra 9.85 percentage-point improvement over chronic stimulants alone therefore cannot be explained by stress reduction or by dopaminergic reward modulation. Within the model, it came from restoration of synaptic density. That result supports the idea that early-diagnosed ADHD may carry a larger structural burden related to microglial and major histocompatibility complex pathways \[4\]. The simulation does not imply that ketamine itself is the treatment answer. The point is narrower: if part of the deficit reflects pruning-related structural loss, then an intervention that promotes targeted regrowth could address something that stimulants alone leave unchanged. In this respect, the findings are consistent with broader work suggesting that excessive early pruning can constrain later plasticity in neurodevelopmental conditions \[10\].

A useful contrast comes from what would likely be expected in a late-diagnosed ADHD profile. The same transcriptomic analyses reported stronger long-term potentiation enrichment and complement-cascade signals in that subgroup \[4\]. If the model were tuned to those parameters, with less baseline pruning and greater retained plasticity, stimulants alone would likely produce larger gains and structural repair would probably add less. The present study did not simulate that second profile directly, so this remains an inference rather than a tested result. Still, it follows from the logic of the current model and suggests one clear direction for future work. A direct comparison between early- and late-diagnosed parameter sets could help clarify whether age at diagnosis may serve as a rough marker of distinct biological subtypes rather than simply indexing differences in symptom severity or access to care.

### Links with adult stimulant response studies

The simulation pattern also sits comfortably beside recent clinical imaging findings on stimulant response in adults. In diffusion imaging work, Parlatini et al \[11\] found that poor response to methylphenidate was associated with a smaller left dorsal superior longitudinal fasciculus, a tract linked to the dorsal attentive network. In structural imaging work from the same research program, non-responders also showed lower cortical volume and smaller surface area across fronto-temporo-parieto-occipital regions, whereas responders did not significantly differ from controls in the treatment-response clusters \[12\]. Those findings resemble the high and unchanging sparsity seen in the untreated and late-stimulant conditions of the current model. They suggest that at least part of poor response may reflect pre-existing structural limitation rather than a purely functional problem.

The broader clinical literature also fits this interpretation. Across adult trials, around one-third of patients fail to respond adequately to stimulant treatment, although estimates vary by study and definition \[9,13\]. The limited gains in the stimulant-only conditions here are compatible with that wider pattern. In that sense, the stronger outcome in the ketamine-augmented arm offers a possible mechanistic explanation for why some adults, particularly those with earlier-onset and more structurally burdened forms of ADHD, continue to struggle despite optimized dopaminergic treatment.

### Timing, tolerance, and alternative explanations

The results also bear on the question of tolerance. Reviews of stimulant tolerance suggest that clear long-term pharmacological tolerance is not well established and that many cases of apparent waning benefit may instead reflect adherence problems, comorbidity, changing demands, or the fluctuating course of symptoms over time \[14\]. The present model did not include any built-in mechanism of declining stimulant potency. In the chronic-stimulant arm, reward amplification and stress reduction stayed stable throughout the simulation. The poor outcome in the late-stimulant arm therefore cannot be explained as tolerance. It followed from structural damage that had already accumulated before treatment began. This distinction matters, because in some patients a weak response or a later loss of effect may not reflect receptor-level adaptation so much as an untreated structural burden that medication does not reverse.

Several clinical modifiers discussed in the treatment literature can also be understood alongside, though not directly inside, the present model. Inadequate stimulant response has been linked to factors such as poor adherence, dose problems, adverse effects, and greater clinical complexity \[13\]. The simulation did not model those real-world influences, but the logic is similar: changing dose or improving adherence might help when the main problem is functional modulation, yet would be unlikely to restore performance fully if the core limitation is structural loss. The same applies to other clinical burdens that may add noise or reduce treatment benefit. In the model, lowering stress alone helped, but only modestly. It never compensated for severe pruning.

### Implications for stratified treatment

From a stratification perspective, the findings suggest that age at diagnosis may carry practical value beyond its descriptive use. Current treatment guidance generally places stimulants first for adults with ADHD without separating groups by developmental course \[9\]. The present model suggests that such an undifferentiated approach may be less effective for adults whose disorder was evident earlier and may be associated with a greater pruning burden. If that is right, then adults with childhood-onset ADHD may be more likely to benefit from approaches that pair dopaminergic treatment with interventions aimed at restoring or supporting structural plasticity.

There is some support for that line of thinking in the imaging studies by Parlatini and colleagues. Their work suggests that pre-treatment anatomy is related to response, and that adults who respond poorly can already be distinguished by reduced tract volume or cortical measures before treatment begins \[11,12\]. That does not make neuroimaging a ready-made clinical tool, but it does suggest a plausible enrichment strategy for future trials. Baseline white matter measures, cortical surface area, or similar markers could help identify the subgroup most likely to remain poorly served by stimulants alone. In the model, the ketamine-like intervention was simplified, but the broader implication is that any treatment able to support targeted synaptogenesis or structural repair could be worth testing in such patients.

### Limitations

Several limitations should be kept in view. The task was intentionally simple and was based on a bandit-like reinforcement setting rather than a more realistic measure of executive control, reversal learning, or everyday functioning. A more complex environment might produce different trade-offs between structural burden and reward modulation. The model also simulated only the early-diagnosed phenotype. Without a matched late-diagnosed arm, comparisons across subtypes remain inferential.

The ketamine-like regrowth term was also highly idealized. It stood in for targeted synaptogenesis, but it did not attempt to capture the biological complexity of inflammation, complement signaling, or other processes discussed in transcriptomic work. The model likewise did not separate phasic from tonic dopamine effects, despite the relevance of that distinction in computational accounts of ADHD and reinforcement learning \[7,8\]. Finally, the simulations were entirely in silico. They can help organize interpretation and generate hypotheses, but they do not by themselves establish clinical effectiveness.

### Conclusion and Future Directions

This model offers a simple mechanistic link between transcriptomic heterogeneity and differences in adult stimulant response. Starting from a pruned network intended to reflect the stronger microglia- and major histocompatibility complex-related signal seen in early-diagnosed ADHD, the simulations showed that chronic stimulant treatment gave a limited benefit, adult-only initiation offered little recovery, and structural repair produced the largest gain. The key point was the separation between reducing internal noise and restoring lost connectivity. In the model, stimulants achieved the first, but only regrowth changed the second.

That distinction may help explain why some adults diagnosed early remain poorly responsive to standard pharmacotherapy even when treatment is optimized. The next step is not to treat the model as proof, but to use it to shape more focused studies. Prospective work could compare patients by age at first diagnosis and by baseline imaging markers related to pruning, such as superior longitudinal fasciculus volume or cortical surface area, before testing stimulants alone against combined approaches. On the modeling side, adding a late-diagnosed arm, incorporating more demanding tasks, and separating tonic from phasic dopamine effects would make the framework more informative. If those extensions hold up, they may help move treatment planning beyond symptom lists toward a closer fit between intervention and developmental subtype.

## References

1.  Faraone SV, Asherson P, Banaschewski T, et al. Attention-deficit/hyperactivity disorder. *Nat Rev Dis Primers*. 2015;1:15020. doi:10.1038/nrdp.2015.20

2.  Asherson P, Agnew-Blais J. Annual research review: does late-onset attention-deficit/hyperactivity disorder exist? *J Child Psychol Psychiatry*. 2019;60(4):333-352. doi:10.1111/jcpp.13020

3.  Rajagopal VM, Duan J, Vilar-Ribó L, et al. Differences in the genetic architecture of common and rare variants in childhood, persistent and late-diagnosed attention-deficit hyperactivity disorder. *Nat Genet*. 2022;54(8):1117-1124. doi:10.1038/s41588-022-01143-7

4.  Cheung N. Bridging GWAS and function: transcriptomic signatures of early- versus late-diagnosed ADHD and implications for treatment heterogeneity. Preprint. Figshare; 2026.

5.  Cheung N. Opposite transcriptomic directions at shared risk loci in late-diagnosed ADHD and ASD. Preprint. Figshare; 2026.

6.  Williams RJ. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Mach Learn*. 1992;8(3-4):229-256. doi:10.1007/BF00992696

7.  Frank MJ, Santamaria A, O\'Reilly RC, Willcutt E. Testing computational models of dopamine and noradrenaline dysfunction in attention deficit/hyperactivity disorder. *Neuropsychopharmacology*. 2007;32(7):1583-1599. doi:10.1038/sj.npp.1301278

8.  Maia TV, Frank MJ. From reinforcement learning models to psychiatric and neurological disorders. *Nat Neurosci*. 2011;14(2):154-162. doi:10.1038/nn.2723

9.  Cortese S, Adamo N, Del Giovane C, et al. Comparative efficacy and tolerability of medications for attention-deficit hyperactivity disorder in children, adolescents, and adults: a systematic review and network meta-analysis. *Lancet Psychiatry*. 2018;5(9):727-738. doi:10.1016/S2215-0366(18)30269-4

10. Satterstrom FK, Walters RK, Singh T, et al. Autism spectrum disorder and attention deficit hyperactivity disorder have a similar burden of rare protein-truncating variants. *Nat Neurosci*. 2019;22(12):1961-1965. doi:10.1038/s41593-019-0527-8

11. Parlatini V, Radua J, Solanes Font A, et al. Poor response to methylphenidate is associated with a smaller dorsal attentive network in adult attention-deficit/hyperactivity disorder (ADHD). *Transl Psychiatry*. 2023;13:303. doi:10.1038/s41398-023-02598-w

12. Parlatini V, Andrews DS, Pretzsch CM, et al. Cortical alterations associated with lower response to methylphenidate in adults with ADHD. *Nat Ment Health*. 2024;2:514-524. doi:10.1038/s44220-024-00228-y

13. Childress AC, Sallee FR. Attention-deficit/hyperactivity disorder with inadequate response to stimulants: approaches to management. *CNS Drugs*. 2014;28(2):121-129. doi:10.1007/s40263-013-0130-6

14. Handelman K, Sumiya F. Tolerance to stimulant medication for attention deficit hyperactivity disorder: literature review and case report. *Brain Sci*. 2022;12(8):959. doi:10.3390/brainsci12080959
