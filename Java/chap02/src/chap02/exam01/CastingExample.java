package chap01.exam01;

public class CastingExample {

	public static void main(String[] args) {
		/*int intValue = 44032;
		char charValue = (char) intValue;
		System.out.println(charValue);*/
		
		int intValue = 44132;
		if (intValue > Byte.MAX_VALUE || intValue < Byte.MIN_VALUE) {
			System.out.println("�ٲ� �� �����ϴ�");
		} else {
			byte byteValue = (byte) intValue;
			System.out.println(byteValue);
		}
		
		
		long longValue = 500;
		intValue = (int) longValue;
		System.out.println(intValue);

	}

}
