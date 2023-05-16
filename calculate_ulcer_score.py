import json

'''
This script read two jsons containing the amount of pixels in the ulcer and the body 
and calculate the ulcer factor by dividing the ulcer pixels by the body pixels
The result is saved to a json file. 
'''

body_json = json.load(open('body_pixels.json'))

ulcer_json = json.load(open('ulcer_pixels.json'))

body_values = []
ulcer_values = []
image_ids = []

ulcer_factor_dict = {}

for b_key, b_pxls in body_json.items():
    body_values.append(b_pxls[0])

for u_key, u_pxls in ulcer_json.items():
        ulcer_values.append(u_pxls[0])
        image_ids.append(u_key)

for i in range(len(ulcer_values)):
        ulcer_factor_dict[image_ids[i]] = ulcer_values[i]/body_values[i]

with open('ulcer_factor.json', 'w') as f:
    json.dump(ulcer_factor_dict, f)

print(ulcer_factor_dict)