
def pretty_time(seconds):
    mins, secs = divmod(seconds, 60)
    print(mins, secs)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    return "{0} days, {1} hrs, {2} mins, {3}s".format(days, hrs, mins, secs)

t = pretty_time(100)
print(t)