"""Image-position proximity bands, never calibrated metric distance."""
def proximity(detection, geometry, cfg):
    static = detection["class_name"] in cfg["static_ground_classes"]
    bottom=geometry["point"][1]
    reliable=not geometry["bottom_clipped"]
    if not reliable:
        band="unknown"
    elif bottom>=cfg["static_danger_y"]:
        band="near"
    elif bottom>=cfg["static_caution_y"]:
        band="middle"
    else:
        band="far"
    return {"policy":"static_ground" if static else "mobile_or_other",
            "band":band,"contact_reliable":reliable,"bottom_y":bottom,
            "distance_m":None,"basis":"image_ground_contact"}
