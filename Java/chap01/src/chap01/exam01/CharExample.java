package chap01.exam01;

public class CharExample {

	public static void main(String[] args) {
		char c1 = 'A';
		String c11 = "AB";
		System.out.println(c1);
		System.out.println(c11);
		
		// �����ڵ� 0~65535 65���� A
		
		char c2 = 65;			
		System.out.println(c2);
		
		char c3 = '\uac00';
		System.out.println(c3);
		
		// b�� �����ڵ� ����?
		char c4 = 'b';
		int unicodeValue = c4;
		System.out.println(unicodeValue);
		// '��'�� �����ڵ� ����? 44035~55203
		char c5 = '��';
		int unicodeValue2 = c5;
		System.out.println(unicodeValue2);

	}

}
