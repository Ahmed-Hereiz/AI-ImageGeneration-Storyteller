# AI-ImageGeneration-Storyteller

## In this project, I have developed two generative AI models:
> - Generating images from text using Stable Diffusion.<br>
> - Generating stories from text using Causal Language Modeling.<br>
> - I've also utilized Streamlit, allowing to effortlessly interact with and explore these remarkable AI creations.<br>
<br>

## Stable-Diffusion model:
<p>The image generation component utilizes <strong>LoRA fine-tuning</strong>, applied on top of a pre-trained "stable-diffusion-v1-4" model. This allows the model to create high-quality images based on textual prompts, enabling users to visualize their ideas with artistic and realistic images.</p>
<br>


### Here are some examples of the model outputs...
### Entering to the prompt : `Alien sitting on table`
<div style="display: flex; justify-content: flex-start;">
    <img src="stablediff_generated_outputs/alien_sitting_on_table1.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/alien_sitting_on_table2.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/alien_sitting_on_table3.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/alien_sitting_on_table4.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/alien_sitting_on_table5.jpg" alt="Image Description" width="190" height="190">
</div>
<br>

### Entering to the prompt : `Alien in desert`
<div style="display: flex; justify-content: flex-start;">
    <img src="stablediff_generated_outputs/alien_in_desert1.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/alien_in_desert2.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/alien_in_desert3.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/alien_in_desert4.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/alien_in_desert5.jpg" alt="Image Description" width="190" height="190">
</div>
<br>

### Entering to the prompt : `Batman wearing captain America suit`
<div style="display: flex; justify-content: flex-start;">
    <img src="stablediff_generated_outputs/batman_wear_captain_america_suit1.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/batman_wear_captain_america_suit2.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/batman_wear_captain_america_suit3.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/batman_wear_captain_america_suit4.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/batman_wear_captain_america_suit5.jpg" alt="Image Description" width="190" height="190">
</div>
<br>

### Entering to the prompt : `Batman pink suit`
<div style="display: flex; justify-content: flex-start;">
    <img src="stablediff_generated_outputs/batman_pink_suit1.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/batman_pink_suit2.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/batman_pink_suit3.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/batman_pink_suit4.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/batman_pink_suit5.jpg" alt="Image Description" width="190" height="190">
</div>
<br>

### Entering to the prompt : `Blue golden Iron man`
<div style="display: flex; justify-content: flex-start;">
    <img src="stablediff_generated_outputs/blue_golden_ironman1.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/blue_golden_ironman2.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/blue_golden_ironman3.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/blue_golden_ironman4.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/blue_golden_ironman5.jpg" alt="Image Description" width="190" height="190">
</div>
<br>

### Entering to the prompt : `Blue cat in jungle`
<div style="display: flex; justify-content: flex-start;">
    <img src="stablediff_generated_outputs/blue_cat_in_jungle1.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/blue_cat_in_jungle2.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/blue_cat_in_jungle3.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/blue_cat_in_jungle4.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/blue_cat_in_jungle5.jpg" alt="Image Description" width="190" height="190">
</div>
<br>

### Entering to the prompt : `Green dog with red eyes`
<div style="display: flex; justify-content: flex-start;">
    <img src="stablediff_generated_outputs/green_dog_with_red_eyes1.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/green_dog_with_red_eyes2.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/green_dog_with_red_eyes3.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/green_dog_with_red_eyes4.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/green_dog_with_red_eyes5.jpg" alt="Image Description" width="190" height="190">
</div>
<br>

### Entering to the prompt : `Green elephant near the sea`
<div style="display: flex; justify-content: flex-start;">
    <img src="stablediff_generated_outputs/green_elephant_near_the_sea1.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/green_elephant_near_the_sea2.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/green_elephant_near_the_sea3.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/green_elephant_near_the_sea4.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/green_elephant_near_the_sea5.jpg" alt="Image Description" width="190" height="190">
</div>
<br>

### Entering to the prompt : `Red polar bear in snow`
<div style="display: flex; justify-content: flex-start;">
    <img src="stablediff_generated_outputs/red_polar_bear_in_snow1.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/red_polar_bear_in_snow2.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/red_polar_bear_in_snow3.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/red_polar_bear_in_snow4.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/red_polar_bear_in_snow5.jpg" alt="Image Description" width="190" height="190">
</div>
<br>

### Entering to the prompt : `Strange creature in the sea`
<div style="display: flex; justify-content: flex-start;">
    <img src="stablediff_generated_outputs/strange_creature_in_sea1.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/strange_creature_in_sea2.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/strange_creature_in_sea3.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/strange_creature_in_sea4.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/strange_creature_in_sea5.jpg" alt="Image Description" width="190" height="190">
</div>
<br>

### Entering to the prompt : `Egypt pyramids in the ice`
<div style="display: flex; justify-content: flex-start;">
    <img src="stablediff_generated_outputs/Egypt_pyramids_in_snow1.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/Egypt_pyramids_in_snow2.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/Egypt_pyramids_in_snow3.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/Egypt_pyramids_in_snow4.jpg" alt="Image Description" width="190" height="190">
    <img src="stablediff_generated_outputs/Egypt_pyramids_in_snow5.jpg" alt="Image Description" width="190" height="190">
</div>
<br><br>

## Story-Generation model:
<p>The story generation functionality is achieved by <strong>full fine-tuning</strong> a pre-trained decoder model "GPT-2" on a <strong>dataset of narrative texts.</strong> The model is trained to generate coherent and engaging stories, leveraging its language understanding capabilities.</p>
<p>Also I added 6 stories genre type to enhance the creativity !!</p>
<br>

### Here are some examples of the model outputs...

### Entering to the prompt : `Alien` and choosing `thriller` to generate the story
<div style="display: flex; justify-content: flex-start;">
    <img src="storymodel_generated_outputs/aliens_thriller_story.png" alt="Image Description" width="900" height="300">
</div>
<br>

### Entering to the prompt : `Robot army` and choosing `horror` to generate the story
<div style="display: flex; justify-content: flex-start;">
    <img src="storymodel_generated_outputs/robot_army_horror_story.png" alt="Image Description" width="900" height="300">
</div>
<br>

### Entering to the prompt : `A young boy` and choosing `superhero` to generate the story
<div style="display: flex; justify-content: flex-start;">
    <img src="storymodel_generated_outputs/young_boy_superhero_story.png" alt="Image Description" width="900" height="300">
</div>
<br>


## Streamlit Framework:
<p>The interactive user interface is developed using Streamlit, a powerful Python library. It enables seamless integration of the AI models with user input and output, creating a user-friendly experience for generating captivating stories and stunning images effortlessly.</p>
<br>

### Here are some screenshots from the app to see how it looks like...

### Entering to the prompt : `The main page`
<div style="display: flex; justify-content: flex-start;">
    <img src="app_screenshots/page_intro.png" alt="Image Description" width="900" height="300">
</div>
<br>

### Entering to the prompt : `After choosing text to image generation`
<div style="display: flex; justify-content: flex-start;">
    <img src="app_screenshots/txt2img.png" alt="Image Description" width="900" height="300">
</div>
<br>

### Entering to the prompt : `After choosing story generation`
<div style="display: flex; justify-content: flex-start;">
    <img src="app_screenshots/txt2story.png" alt="Image Description" width="900" height="300">
</div>
<br>


# Finally more will be added...
