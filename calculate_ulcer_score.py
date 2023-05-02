import json

'''
This file read two jsons containing the amount of pixels in the ulcer and the body 
and calculate the ulcer factor by dividing the ulcer pixels by the body pixels
The result is saved to a json file. 
'''

body_json = json.load(open('body_pixels.json'))

ulcer_json = json.load(open('ulcer_pixels.json'))

ulcer_factor_dict = {}


for b_key, b_pxls in body_json.items():
        b_key = b_key.split('_')[1]
        for u_key, u_pxls in ulcer_json.items():
                u_key = u_key.split('_')[1]
                if str(u_key) in str(u_key):
                        ulcer_factor_dict[b_key] = u_pxls[0]/b_pxls[0]
                        print( 'u pixels' ,u_pxls[0])
                        print( ' b  pixels' , b_pxls[0])
                        print( 'relation ', u_pxls[0]/b_pxls[0])
                        break
                        

with open('ulcer_factor.json', 'w') as f:
    json.dump(ulcer_factor_dict, f)