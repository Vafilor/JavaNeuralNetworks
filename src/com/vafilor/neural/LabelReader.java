package com.vafilor.neural;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Andrey Melnikov on 5/14/2016.
 */
public class LabelReader
{
    public static List<Matrix> convertData(Path path)
    {
        InputStream input = null;

        try {

            byte[] number = new byte[4];

            input = Files.newInputStream(path);

            input.read(number);

            int magicNumber = getInteger(number);

            input.read(number);

            int labels = getInteger(number);

            List<Matrix> labelsAsVectors = new ArrayList<>(labels);

            //Output is number from 0 - 9
            Matrix label = new Matrix(10, 1);

            byte[] numberValues = new byte[labels];
            input.read(numberValues);

            label.setEntry(numberValues[0], 0, 1);
            labelsAsVectors.add(label);

            for(int i = 1; i < numberValues.length; i++)
            {
                label = new Matrix(10, 1);
                label.setEntry(numberValues[i], 0, 1);
                labelsAsVectors.add(label);
            }

            return labelsAsVectors;

        } catch(IOException exception)
        {
            System.out.format("Error in reading file:" + exception.getMessage());
            return new ArrayList<>();
        } finally
        {
            if( null != input )
            {
                try {
                    input.close();
                }
                catch (IOException exception)
                {
                    System.err.format("Failed to close input stream for %s. Error %s", path, exception.getMessage());
                }
            }
        }
    }

    private static int getInteger(byte[] number)
    {
        return ByteBuffer.wrap(number).getInt();
    }
}
