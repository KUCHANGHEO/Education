package chap01.exam01;

public class FromIntToFloat {

	public static void main(String[] args) {
		int num1 = 1234567890;
		int num2 = 123456;
		
		float num3 = num2;
		/*0.1234*10^4
		4    * 1234   
		
		 31Ä­ : 21¾ï
		21Ä­
		0.1234567890 *10^9
		   9 *123456
		*/
		num2 = (int) num3;
		
		int result = num1 - num2;
		System.out.println(result);

	}

}
