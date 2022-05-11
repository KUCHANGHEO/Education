package chap06.exam01;

public class Calculator {
	//field
	//constuctor
	// method
	
static	void powerOn() { //static 메소드 , 정적 메소드
		System.out.println("전원을 켭니다.");
		}
	
	int plus(int x, int y) {
		int result = x + y;
		return result;
	}
	
	double divide(int x, int y) {
		double result = (double)x / (double)y;
		return result;
	}
	
	void powerOff() {
		System.out.println("전원을 끕니다.");
	}
}
