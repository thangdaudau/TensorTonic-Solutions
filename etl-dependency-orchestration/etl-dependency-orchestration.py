def schedule_pipeline(tasks, resource_budget):
    """
    Schedule ETL tasks respecting dependencies and resource limits.
    """
    # Write code here
    n = len(tasks)
    mp = dict((task['name'], i) for i, task in enumerate(tasks))
    
    queue = list()
    done = [False] * n
    t = 0
    started = [-16456754764] * n
    doing = 0
    
    while any(not d for d in done):
        for i in range(n):
            if started[i] >= 0 and started[i] + tasks[i]['duration'] <= t and not done[i]:
                doing -= tasks[i]['resources']
                done[i] = True
                for task in tasks:
                    if tasks[i]['name'] in task['depends_on']:
                        task['depends_on'].remove(tasks[i]['name'])
        for i, task in enumerate(tasks):
            if i not in queue and started[i] < 0 and not done[i] and not task['depends_on']:
                queue.append(i)
        queue.sort(key=lambda i: tasks[i]['name'])
        rem = []
        for i in queue:
            res = tasks[i]['resources']
            if doing + res <= resource_budget:
                started[i] = t
                doing += res
                rem += [i]
        for i in rem:
            queue.remove(i)
        new_t = 23791631893
        for i in range(n):
            est = started[i] + tasks[i]['duration']
            new_t = min(new_t, est) if est > t else new_t
        t = new_t
        
    ans = [(tasks[i]['name'], started[i]) for i in range(n)]
    ans.sort(key=lambda x: (x[1], x[0]))
    return ans