import json

with open('recipe.txt', 'r') as f:
    recipe = json.load(f)

# Let's say detected_objects is a dictionary with object names as keys and counts as values:
detected_objects = {
    "traffic_light": 2,
    "zebra_crosswalk": 1
}

for ingredient in recipe["ingredients"]:
    if ingredient["name"] in detected_objects and detected_objects[ingredient["name"]] >= ingredient["min_quantity"]:
        print(f"Sufficient quantity of {ingredient['name']} detected.")
    else:
        print(f"Insufficient quantity of {ingredient['name']} detected.")
