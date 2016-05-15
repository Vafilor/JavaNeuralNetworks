package com.vafilor.neural;

import java.util.List;
import java.util.ArrayList;

/**
 * Created by Andrey Melnikov on 5/14/2016.
 *
 * Helper class to time how long something took.
 * Basic usage:
 *  mark() before method call.
 *  mark() after method call.
 *  Call getLastMarksElapsedTimeInSeconds() to get elapsed time in seconds.
 */
public class Timer {
    private List<Long> timestamps;

    public Timer() {
        this.timestamps = new ArrayList<>();
    }

    public void mark() {
        this.timestamps.add(System.currentTimeMillis());
    }

    public long getTotalElapsedTime()
    {
        if(this.timestamps.size() < 2)
        {
            throw new IllegalStateException("Less than 2 timestamps recorded. Can't get elapsed time.");
        }

        return this.timestamps.get(this.timestamps.size() - 1) - this.timestamps.get(0);
    }

    public long getLastMarksElapsedTime()
    {
        if(this.timestamps.size() < 2)
        {
            throw new IllegalStateException("Less than 2 timestamps recorded. Can't get elapsed time.");
        }

        return this.timestamps.get(this.timestamps.size()-1) - this.timestamps.get(this.timestamps.size() - 2);
    }

    public double getTotalElapsedTimeInSeconds()
    {
        return this.getTotalElapsedTime() / 1000.0;
    }

    public double getLastMarksElapsedTimeInSeconds()
    {
        return this.getLastMarksElapsedTime() / 1000.0;
    }

    public void clear()
    {
        this.timestamps.clear();
    }
}
