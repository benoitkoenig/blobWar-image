import json

images_data = []
statuses = {
    "normal": 0,
    "hat": 1,
    "ghost": 2,
}

with open('data.json') as json_file:  
    data = json.load(json_file)
    for data_index in range(10000):
        picture_data = data[str(data_index)]
        blob_id = None
        blob_count = 0
        all_blobs = picture_data["army"] + picture_data["enemy"]
        for i in range(6):
            b = all_blobs[i]
            if (b["alive"] == True):
                blob_count += 1
                blob_id = i
        if blob_count == 1:
            if blob_id >= 3:
                is_enemy = 1
            else:
                is_enemy = 0
            blob = all_blobs[blob_id]
            status = statuses[blob["status"]]
            label = is_enemy * 3 + status
            images_data.append((data_index, label))
