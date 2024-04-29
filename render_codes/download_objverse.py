import objaverse

# Download lvis subset of Objverse-1.0 only
# Modify BASE_PATH in __init__.py of objverse

uids = objaverse.load_uids()

lvis_annotations = objaverse.load_lvis_annotations()

lvis_uids_all = []
for k, v in lvis_annotations.items():
    lvis_uids_all.extend(v)

lvis_objs = objaverse.load_objects(uids=lvis_uids_all, download_processes=16)

# import trimesh
# trimesh.load(list(lvis_objs.values())[0]).show()