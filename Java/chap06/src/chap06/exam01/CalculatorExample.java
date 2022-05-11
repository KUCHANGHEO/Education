package chap06.exam01;

public class CalculatorExample {

	public static void main(String[] args) {
		
		Calculator cal = new Calculator();
		cal.powerOn();
		int num1 = cal.plus(1, 2);
		double num2 = cal.divide(1, 2);
		System.out.println(num1 +" "+ num2);
		cal.powerOff();
		
		Calculator.powerOn();
		
		double result = Math.random();
		System.out.println(result);
		System.out.println(Math.PI);
		System.out.println(Math.abs(-7));

	}

}
/*
int divide(int x, int y) {
	
}

diivide(1,2)

int divide(parameter = arguument) {
			매개변수		인자 인수
}
*/