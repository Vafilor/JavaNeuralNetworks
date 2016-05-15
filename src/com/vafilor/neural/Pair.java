package com.vafilor.neural;

/**
 * Created by Andrey Melnikov on 5/13/2016.
 */
public class Pair<K, V>
{
    private K firstElement;
    private V secondElement;

    public Pair(K firstElement, V secondElement)
    {
        this.firstElement = firstElement;
        this.secondElement = secondElement;
    }

    public K getFirstElement()
    {
        return this.firstElement;
    }

    public V getSecondElement()
    {
        return this.secondElement;
    }

    @Override
    public String toString()
    {
        return "(" + this.firstElement + "," + this.secondElement + ")";
    }
}
