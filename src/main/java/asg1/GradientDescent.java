package asg1;

/**
 * Author: baojianfeng
 * Date: 2018-01-26
 * Description: implementation of gradient descent
 */
public class GradientDescent {
    private double[] x; // attribute
    private double[] y; // real results
    private double theta0; // weight0 parameter
    private double theta1; // weight1 parameter
    private double step; // learning rate
    private double[] x0;
    private double[] hypotheses; // hypotheses

    public GradientDescent(double[] x, double[] y, double theta0, double theta1, double step) {
        this.x = x;
        this.y = y;
        this.theta0 = theta0;
        this.theta1 = theta1;
        this.step = step;
        hypotheses = new double[x.length];

        x0 = new double[x.length];
        initializeX0(x.length);
    }

    /**
     * initialize x0 attribute, default is 1
     * @param length sample length
     */
    private void initializeX0(int length) {
        for (int i = 0; i < length; i++)
            x0[i] = 1.0;
    }

    /**
     * calculate error
     * @return error
     */
    public double calError() {
        calHypotheses(theta0, theta1);
        double error = 0.0;
        for (int i = 0; i < x.length; i++)
            error += Math.pow((hypotheses[i] - y[i]), 2);

        return error / (2 * x.length);
    }

    /**
     * update theta value using gradient descent algorithm
     */
    public void updateTheta() {
        theta0 = theta0 - step * calDerivative(x0);
        theta1 = theta1 - step * calDerivative(x);
    }

    /**
     * calculate derivative
     * @param x x attribute
     * @return derivative of specific x attribute
     */
    private double calDerivative(double[] x) {
        double sum = 0.0;
        for (int i = 0; i < hypotheses.length; i++)
            sum += (hypotheses[i] - y[i]) * x[i];

        return sum / hypotheses.length;
    }

    /**
     * calculate hypotheses
     * @param theta0 theta0
     * @param theta1 theta1
     */
    private void calHypotheses(double theta0, double theta1) {
        int length = x.length;
        for (int i = 0; i < length; i++)
            hypotheses[i] = theta0 + theta1 * x[i];

    }

    /**
     * get the theta0 value
     * @return theta0
     */
    public double getTheta0() {
        return theta0;
    }

    /**
     * get the theta1 value
     * @return theta1
     */
    public double getTheta1() {
        return theta1;
    }

    public static void main(String[] args) {
        double[] x = new double[]{3, 1, 0, 4};
        double[] y = new double[]{2, 2, 1, 3};
        double theta0 = 0.0;
        double theta1 = 1.0;
        double step = 0.1; // here set the learning rate to 0.1
        GradientDescent gd = new GradientDescent(x, y, theta0, theta1, step);
        double error = gd.calError();
        System.out.print("initial error is: " + error + " ");
        System.out.print("initial theta0 is: " + theta0 + " ");
        System.out.println("initial theta1 is: " + theta1);
        for (int i = 0; i < 5; i++) {
            gd.updateTheta();
            error = gd.calError();
            System.out.print("error is: " + error + " ");
            System.out.print("theta0 is: " + gd.getTheta0() + " ");
            System.out.println("theta1 is: " + gd.getTheta1() + " ");
        }
    }
}
