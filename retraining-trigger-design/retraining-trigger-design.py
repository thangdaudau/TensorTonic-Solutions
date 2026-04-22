def retraining_policy(daily_stats, config):
    """
    Decide which days to trigger model retraining.
    """
    # Write code here
    ans = [0]
    for s in daily_stats:
        if s["drift_score"] > config["drift_threshold"] \
            or s["performance"] < config["performance_threshold"] \
            or s["day"] - ans[-1] >= config["max_staleness"]:
            if config["budget"] >= config["retrain_cost"] \
                and (ans[-1] == 0 or s["day"] - ans[-1] >= config["cooldown"]):
                ans += [s["day"]]
                config["budget"] -= config["retrain_cost"]
    return ans[1:]