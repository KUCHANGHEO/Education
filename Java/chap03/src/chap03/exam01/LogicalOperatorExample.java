package chap01.exam02;

public class LogicalOperatorExample {

	public static void main(String[] args) {
		int charCode = 'A';
		System.out.println(charCode); //65
		
		if((charCode >= 65) & (charCode<=90)) {
			System.out.println("대문자");
		}
		if((charCode >= 97) && (charCode<=122)) { 
			// &&는 앞에것이 거짓이면 바로 거짓으로 처리한다
			System.out.println("소문자");
		}
		if((charCode < 48) && (charCode >57)) {
			System.out.println("0~9 숫자 이군요");
		}
		
		int value = 6;
		
		if((value%2 == 0) | (value%3 == 0)) {
			System.out.println("2 또는 3의 배수 이군요");
		}
		
		if((value%2 == 0) || (value%3 == 0)) {
			// ||는 앞에가 참이면 참으로 처리해버린다
			System.out.println("2 또는 3의 배수 이군요");
		}


	}

}
