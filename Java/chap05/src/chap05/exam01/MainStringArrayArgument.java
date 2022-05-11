package chap05.exam01;

public class MainStringArrayArgument {

	public static void main(String[] args) {
		String strNum1 = args[0];
		String strNum2 = args[1];
		
		int num1 = Integer.parseInt(strNum1);
		int num2 = Integer.parseInt(strNum2);
		
		
		System.out.println(strNum1 + strNum2);
		System.out.println(num1 + num2);
	}


}


