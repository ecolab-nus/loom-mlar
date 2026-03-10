enum Schedule {
    Parallel(Schedule),
    Sequential(Schedule),
    Op
}