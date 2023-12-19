using System;

class Program
{
    static void Main()
    {
        Console.WriteLine("Enter the number of rows for matrix A:");
        int rowsA = int.Parse(Console.ReadLine());

        Console.WriteLine("Enter the number of columns for matrix A:");
        int colsA = int.Parse(Console.ReadLine());

        double[,] A = new double[rowsA, colsA];

        Console.WriteLine("Enter the values for matrix A:");

        for (int i = 0; i < rowsA; i++)
        {
            for (int j = 0; j < colsA; j++)
            {
                Console.Write($"A[{i},{j}]: ");
                A[i, j] = double.Parse(Console.ReadLine());
            }
        }

        Console.WriteLine("Enter the values for vector B:");

        double[] B = new double[rowsA];

        for (int i = 0; i < rowsA; i++)
        {
            Console.Write($"B[{i}]: ");
            B[i] = double.Parse(Console.ReadLine());
        }

        int iters = 2000;
        double eps = 1e-4;

        double[] result = GradientDescent(A, B, iters, eps);

        // Print the result or use it as needed
        Console.WriteLine("Result:");
        foreach (var value in result)
        {
            Console.Write(value + " ");
        }
    }

    static double[] GradientDescent(double[,] A, double[] B, int iters = 2000, double eps = 1e-4)
    {
        // Initialize the variable vector x with zeros
        double[] x = new double[B.Length];

        for (int iter = 0; iter < iters; iter++)
        {
            // Calculate the predicted values
            double[] predictions = MatrixVectorProduct(A, x);

            // Calculate the error
            double[] error = VectorSubtraction(predictions, B);

            double[] R = VectorSubtraction(B, MatrixVectorProduct(A, x));

            double lambd = DotProduct(R, R) / DotProduct(MatrixVectorProduct(A, R), R);

            // Calculate the gradient
            double[] gradient = ScalarMultiplication(2, R);

            // Update the variable vector using the gradient
            x = VectorAddition(x, ScalarMultiplication(lambd, R));

            // Check for convergence
            if (Norm(gradient) < eps)
            {
                Console.WriteLine($"Converged in {iter + 1} epochs.");
                break;
            }
        }

        return x;
    }

    static double[] MatrixVectorProduct(double[,] matrix, double[] vector)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);

        double[] result = new double[rows];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i] += matrix[i, j] * vector[j];
            }
        }

        return result;
    }

    static double[] VectorSubtraction(double[] vector1, double[] vector2)
    {
        int length = vector1.Length;

        double[] result = new double[length];

        for (int i = 0; i < length; i++)
        {
            result[i] = vector1[i] - vector2[i];
        }

        return result;
    }

    static double DotProduct(double[] vector1, double[] vector2)
    {
        int length = vector1.Length;
        double result = 0;

        for (int i = 0; i < length; i++)
        {
            result += vector1[i] * vector2[i];
        }

        return result;
    }

    static double[] ScalarMultiplication(double scalar, double[] vector)
    {
        int length = vector.Length;
        double[] result = new double[length];

        for (int i = 0; i < length; i++)
        {
            result[i] = scalar * vector[i];
        }

        return result;
    }

    static double[] VectorAddition(double[] vector1, double[] vector2)
    {
        int length = vector1.Length;
        double[] result = new double[length];

        for (int i = 0; i < length; i++)
        {
            result[i] = vector1[i] + vector2[i];
        }

        return result;
    }

    static double Norm(double[] vector)
    {
        double sum = 0;

        foreach (var value in vector)
        {
            sum += value * value;
        }

        return Math.Sqrt(sum);
    }
}
