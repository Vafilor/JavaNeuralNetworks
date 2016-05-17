package com.vafilor.neural;

import java.util.function.DoubleUnaryOperator;

/**
 * Created by Andrey Melnikov on 5/9/16.
 *
 * Represents a Mathematical Matrix.
 */
public class Matrix {

    //TODO write unit tests
    //TODO rearrange methods so they flow public to private and similar methods are together (e.g. add/addInto)
    private int rows;
    private int columns;
    private double[][] entries;

    public Matrix(double[][] entries)
    {
        this.setRows(entries.length);
        this.setColumns(entries.length);

        this.entries = new double[this.rows][this.columns];

        for(int i = 0; i < entries.length; i++)
        {
            System.arraycopy(entries[i], 0, this.entries[i], 0, entries.length);
        }
    }

    public Matrix(int rows, int columns)
    {    
        this.setRows(rows);
        this.setColumns(columns);

        this.entries = new double[rows][columns];
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public double getEntry(int row, int column)
    {
        return this.entries[row][column];
    }

    public void setEntry(int row, int column, double value)
    {
        this.entries[row][column] = value;
    }

    public Matrix add(Matrix that)
    {
        if( !Matrix.sameMatrixSize(this, that) )
        {
            throw new IllegalArgumentException("Matrix addition not defined for input matrices\n:" + this.getDimensions() + " x " + that.getDimensions());
        }

        Matrix copy = new Matrix(this.rows, this.columns);

        for(int i = 0; i < this.rows; i++)
        {
            for (int j = 0; j < this.columns; j++) {
                copy.setEntry(i, j, this.entries[i][j] + that.entries[i][j]);
            }
        }

        return copy;
    }

    public Matrix addInto(Matrix that)
    {
        if( !Matrix.sameMatrixSize(this, that) )
        {
            throw new IllegalArgumentException("Matrix addition not defined for input matrices\n:" + this.getDimensions() + " x " + that.getDimensions());
        }
    
        for(int i = 0; i < this.rows; i++)
        {
            for (int j = 0; j < this.columns; j++) {
                this.entries[i][j] = this.entries[i][j] + that.entries[i][j];
            }
        }

        return this;
    }

    public Matrix subtractInto(Matrix that)
    {
    	if( !Matrix.sameMatrixSize(this, that) )
        {
            throw new IllegalArgumentException("Matrix subtraction not defined for input matrices\n:" + this.getDimensions() + " x " + that.getDimensions());
        }
    
        for(int i = 0; i < this.rows; i++)
        {
            for (int j = 0; j < this.columns; j++) {
                this.entries[i][j] = this.entries[i][j] - that.entries[i][j];
            }
        }

        return this;
    }

    public Matrix subtract(Matrix that)
    {
        if( !Matrix.sameMatrixSize(this, that) )
        {
            throw new IllegalArgumentException("Matrix subtraction not defined for input matrices\n:" + this.getDimensions() + " x " + that.getDimensions());
        }
    
        Matrix copy = new Matrix(this.rows, this.columns);

        for(int i = 0; i < this.rows; i++)
        {
            for (int j = 0; j < this.columns; j++) {
                copy.setEntry(i, j, this.entries[i][j] - that.entries[i][j]);
            }
        }

        return copy;
    }

    public Matrix multiply(Matrix that)
    {
        if(this.columns != that.rows)
        {
            throw new IllegalArgumentException("Matrix multiplication not defined for input matrices.\n" + this.getDimensions() + " x " + that.getDimensions() );
        }

        int newRows = this.rows;
        int newColumns = that.columns;

        Matrix product = new Matrix(newRows, newColumns);

        double newValue = 0.0;

        for(int i = 0; i < newRows; i++)
        {
            for(int j = 0; j < newColumns; j++)
            {
                for(int k = 0; k < this.getColumns(); k++)
                {
                    newValue += this.entries[i][k] * that.entries[k][j];
                }

                product.entries[i][j] = newValue;

                newValue = 0.0;
            }
        }

        return product;
    }

    /**
     * Multiplies each entry of this matrix by each entry of that matrix, entry by entry.
     * Result(i,j) = this(i,j) * that(i,j)
     * @param that
     * @return
     */
    public Matrix multiplyEntries(Matrix that)
    {
        if(this.getRows() != that.getRows() || this.getColumns() != that.getColumns())
        {
            throw new IllegalArgumentException("Matrix entry multplication not defined for input matrices\n:" + this.getDimensions() + " x " + that.getDimensions());
        }

        Matrix result = new Matrix(this.rows, this.columns);

        for (int i = 0; i < this.getRows(); i++) {
            for (int j = 0; j < this.getColumns(); j++) {
                result.setEntry(i, j, this.entries[i][j] * that.entries[i][j]);
            }
        }

        return result;
    }

    public Matrix multiplyEntriesInto(Matrix that)
    {
        if(this.getRows() != that.getRows() || this.getColumns() != that.getColumns())
        {
            throw new IllegalArgumentException("Matrix entry multplication not defined for input matrices\n:" + this.getDimensions() + " x " + that.getDimensions());
        }
        
        for (int i = 0; i < this.getRows(); i++) {
            for (int j = 0; j < this.getColumns(); j++) {
                this.entries[i][j] = this.entries[i][j] * that.entries[i][j];
            }
        }

        return this;
    }

    public Matrix scale(double scalar)
    {
        Matrix copy = new Matrix(this.rows, this.columns);

        for(int i = 0; i < this.rows; i++)
        {
            for (int j = 0; j < this.columns; j++) {
                copy.setEntry(i, j, this.entries[i][j] * scalar);
            }
        }

        return copy;
    }

    public Matrix scaleInto(double scalar)
    {
        for(int i = 0; i < this.rows; i++)
        {
            for (int j = 0; j < this.columns; j++)
            {
                this.entries[i][j] = this.entries[i][j] * scalar;
            }
        }

        return this;
    }

    public Matrix applyFunction(DoubleUnaryOperator operator)
    {
        Matrix copy = new Matrix(this.rows, this.columns);

        for(int i = 0; i < this.rows; i++)
        {
            for (int j = 0; j < this.columns; j++) {
                copy.setEntry(i, j,  operator.applyAsDouble(this.entries[i][j]));
            }
        }

        return copy;
    }

    public Matrix applyFunctionInto(DoubleUnaryOperator operator)
    {
        for(int i = 0; i < this.rows; i++)
        {
            for (int j = 0; j < this.columns; j++) {
                this.entries[i][j] = operator.applyAsDouble(this.entries[i][j]);
            }
        }

        return this;
    }

    public Matrix transpose()
    {
        Matrix transpose = new Matrix(this.getColumns(), this.getRows());

        for (int i = 0; i < this.getRows(); i++) {
            for (int j = 0; j < this.getColumns(); j++) {
                transpose.setEntry(j, i, this.entries[i][j]);
            }
        }

        return transpose;
    }

    public Pair<Integer, Integer> getDimensions()
    {
        return new Pair<Integer, Integer>(this.rows, this.columns);
    }

    @Override
    public String toString()
    {
        StringBuilder builder = new StringBuilder();

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.columns; j++) {
                builder.append(this.entries[i][j]);
                builder.append(" ");
            }
            builder.append("\n");
        }

        return builder.toString();
    }

    private void setRows(int rows)
    {
        if(rows < 1)
        {
            throw new IllegalArgumentException("rows < 0");
        }

        this.rows = rows;
    }

    private void setColumns(int columns)
    {
        if(columns < 1)
        {
            throw new IllegalArgumentException("columns < 0 ");
        }

        this.columns = columns;
    }
    
    public static boolean sameMatrixSize(Matrix a, Matrix b)
    {
		return (a.rows != b.rows) || (a.columns != b.columns);
    }
}
