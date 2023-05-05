import random
sunglass_shapes = ['Aviator', 'Round', 'Square', 'Oval', 'Cat-eye', 'Butterfly', 'Rectangle', 'Sport', 'Wayfarer', 'Clubmaster']
sunglass_colors = ['Black', 'Brown', 'Gray', 'Silver', 'Gold', 'Rose Gold', 'Blue', 'Green', 'Red', 'Purple']
glass_colors = ['Black','Blue', 'Green', 'Red']

def match_glass(gender, face_ratio, eye_color, nose_size):
    shape = random.randint(0, len(sunglass_shapes)-1)
    color = random.randint(0, len(sunglass_colors)-1)
    glass_color = random.randint(0,len(glass_colors)-1)

    return sunglass_colors[color], sunglass_shapes[shape], glass_colors[glass_color]