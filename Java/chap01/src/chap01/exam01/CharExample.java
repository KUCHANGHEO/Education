package chap01.exam01;

public class CharExample {

	public static void main(String[] args) {
		char c1 = 'A';
		String c11 = "AB";
		System.out.println(c1);
		System.out.println(c11);
		
		// 유니코드 0~65535 65번은 A
		
		char c2 = 65;			
		System.out.println(c2);
		
		char c3 = '\uac00';
		System.out.println(c3);
		
		// b의 유니코드 값은?
		char c4 = 'b';
		int unicodeValue = c4;
		System.out.println(unicodeValue);
		// 'ㄱ'의 유니코드 값은? 44035~55203
		char c5 = 'ㄱ';
		int unicodeValue2 = c5;
		System.out.println(unicodeValue2);

	}

}
