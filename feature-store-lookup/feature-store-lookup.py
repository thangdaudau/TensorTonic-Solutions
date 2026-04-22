def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    # Write code here
    ans = []
    for request in requests:
        user_id = request["user_id"]
        online_features = request["online_features"]
        ans += [{**feature_store.get(user_id, defaults), **online_features}]
    return ans