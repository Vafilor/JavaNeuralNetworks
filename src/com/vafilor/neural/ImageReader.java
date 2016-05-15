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
 *
 * Reads the test training images and converts them into input vectors.
 */
public class ImageReader
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

            int images = getInteger(number);

            List<Matrix> imagesAsVectors = new ArrayList<>(images);


            input.read(number);

            int rows = getInteger(number);

            input.read(number);

            int columns = getInteger(number);

            Matrix image = new Matrix(rows * columns, 1);

            byte[] pixelValue = new byte[1];

            int pixelsRead = 0;

            while( input.read(pixelValue) != -1)
            {
                image.setEntry(pixelsRead, 0, (int)(pixelValue[0] & 0xFF));

                pixelsRead++;

                if(pixelsRead == image.getRows())
                {
                    imagesAsVectors.add(image);
                    image = new Matrix(rows * columns, 1);
                    pixelsRead = 0;
                }
            }

            return imagesAsVectors;

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
