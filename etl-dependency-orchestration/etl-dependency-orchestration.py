def schedule_pipeline(tasks, resource_budget):
    """
    Schedule ETL tasks respecting dependencies and resource limits.
    """
    # Write code here
    inf = 1 << 64
    n = len(tasks)
    get = lambda s: lambda d: d[s]
    name = list(map(get('name'), tasks))
    duration = list(map(get('duration'), tasks))
    res = list(map(get('resources'), tasks))
    depend = list(map(get('depends_on'), tasks))
    
    queue = list()
    t = 0
    ready = [False] * n
    started = [-inf] * n
    doing = 0
    done = [False] * n
    
    while any(not d for d in done):
        
        for i in range(n):
            if started[i] != -inf and started[i] + duration[i] <= t and not done[i]:
                doing -= res[i]
                done[i] = True
                for j in range(n):
                    if name[i] in depend[j]:
                        depend[j].remove(name[i])
        for i in range(n):
            if not ready[i] and not depend[i]:
                ready[i] = True
                queue.append(i)
        
        queue.sort(key=lambda i: name[i])
        rem = []
        for i in queue:
            if doing + res[i] <= resource_budget:
                started[i] = t
                doing += res[i]
                rem += [i]
        for i in rem:
            queue.remove(i)
            
        new_t = inf
        for i in range(n):
            est = started[i] + duration[i]
            new_t = min(new_t, est) if est > t else new_t
        t = new_t
        
    ans = [(tasks[i]['name'], started[i]) for i in range(n)]
    ans.sort(key=lambda x: (x[1], x[0]))
    return ans