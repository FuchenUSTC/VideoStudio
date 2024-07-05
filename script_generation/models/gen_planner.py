base_prompt = '''
Given a single text prompt, you need to envision a multi-scene video by generating a sequence of stepwise prompts to describe the text prompt. For each step, you also need to generate the set of entities needed and describe the background scene where the video should occur. Related steps should maintain similar entities and background scenes. Before you write each stepwise description, you must follow these instructions:
1. Each step prompt must contain only a single motion or action.
2. Each step prompt must include all relevant objects and describe the environment scene.
3. Make sure each step prompt must be easy described by a shot video cilp of 4 seconds at 4fps.
4. Gives 4 steps in total.
5. The output must not contain the word "camera".
6. The output must not contain the word "same".

Use the following format: [step_1: [prompt, entities, background]; step_2: [prompt, entities, background]; … step_n: [prompt, entities, background]]

The example of output:
[step_1: [A woman walks along a village road; entities: woman; background: a picturesque coastal village with quaint houses and a view of the sea]]
[step_2: [The woman's hair gently blows in the sea breeze; entities: woman; background: a picturesque coastal village with quaint houses and a view of the sea]]
[step_3: [The woman carries a basket filled with colorful flowers; entities: woman, basket, colorful flowers; background: a picturesque coastal village with quaint houses and a view of the sea]]
...

Provide stepwise prompts for the text prompt:
'''

base_history = [
{'role': 'user', 'content': base_prompt + 'A skilled chef is cutting roasted duck and putting it on a plate, A close-up of the cut roasted duck'},
{'role': 'assistant', 'content': '''
[step_1: [The chef slices the roasted duck with precision; entities: chef, roasted duck; background: a professional kitchen with stainless steel countertops and utensils]]
[step_2: [The knife glides through the succulent roasted duck meat; entities: chef, roasted duck, knife; background: a professional kitchen with stainless steel countertops and utensils]]
[step_3: [The chef delicately places the sliced roasted duck onto a white porcelain plate; entities: chef, sliced roasted duck, white porcelain plate; background: a professional kitchen with stainless steel countertops and utensils]]
[step_4: [A close-up shot captures the beautifully arranged sliced roasted duck on the plate; entities: sliced roasted duck, white porcelain plate; background: a professional kitchen with stainless steel countertops and utensils]]
'''},
{'role': 'user', 'content': base_prompt + 'A happy elderly couple is riding a tandem bicycle'},
{'role': 'assistant', 'content': '''
[step_1: [The elderly couple mounts the tandem bicycle; entities: elderly couple, tandem bicycle; background: a scenic park with lush greenery and a winding path]]
[step_2: [The breeze rustles their hair as they navigate a gentle slope; entities: elderly couple, tandem bicycle; background: a scenic park with lush greenery and a winding path]]
[step_3: [The couple waves at the children, their laughter filling the air; entities: elderly couple, tandem bicycle, group of children, colorful balloons; background: a scenic park with lush greenery and a winding path]]
[step_4: [They ride into the sunset, their silhouettes fading into the horizon; entities: elderly couple, tandem bicycle; background: a scenic park with lush greenery, a winding path, a small bridge, and a sunset-lit sky]]
'''},
]