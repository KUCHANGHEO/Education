package chap01.exam02;

import java.util.Iterator;

public class AccuracyExample {

	public static void main(String[] args) {
		int apple1 = 1;
		double pieceUnit1 = 0.1;
		int number1 = 7;
		
		double result1 = apple1 - number1 * pieceUnit1;
		
		System.out.println("��� �Ѱ����� ");
		System.out.println("0.7 ������ ����, ");
		System.out.println(result1 + "������ ���´�.");
		
		// 0.1�� ǥ������ ���Ѵ�
		double sum = 0;
		for (int i = 0; i <10; i ++) {
			sum += pieceUnit1;
		}
		System.out.println(sum);
		
		int apple2 = 1;
		int totalPieces = apple2 * 10;
		int number2 = 7;
		int temp = totalPieces - number2;
		
		double result2 = temp/10.0;
		
		System.out.println("��� �Ѱ����� ");
		System.out.println("0.7 ������ ����, ");
		System.out.println(result2 + "������ ���´�.");

	}

}
