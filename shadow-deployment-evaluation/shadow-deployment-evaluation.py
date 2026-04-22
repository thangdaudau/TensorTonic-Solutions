import numpy as np

def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """
    # Write code here
    n = len(production_log)
    get = lambda s: lambda d: d[s]
    actual = np.array(list(map(get('actual'), production_log)))
    pred_prod = np.array(list(map(get('prediction'), production_log)))
    pred_shad = np.array(list(map(get('prediction'), shadow_log)))

    acc_prod = np.sum(pred_prod == actual).item() / n
    acc_shad = np.sum(pred_shad == actual).item() / n
    acc_gain = acc_shad - acc_prod

    n95 = (95 * n + 99) // 100 - 1
    p95 = np.partition(list(map(get('latency_ms'), shadow_log)), n95)[n95].item()

    agree_rate = (pred_prod == pred_shad).sum().item() / n

    promote = acc_gain >= criteria['min_accuracy_gain'] \
                and p95 <= criteria['max_latency_p95'] \
                and agree_rate >= criteria['min_agreement_rate']

    return {'promote': promote, 'metrics': {'shadow_accuracy': acc_shad, 'production_accuracy': acc_prod, 'accuracy_gain': acc_gain, 'shadow_latency_p95': p95, 'agreement_rate': agree_rate}}