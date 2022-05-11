package chap06.exam01;

public class Car {
	// 필드 = 변수
	static	String company = "현대자동차";
			String model = "그랜저";
			String color = "검정";
			int maxSpeed = 350;
			int porductYear;
			int speed;
			int gas;
	
	// 생성자
	// 오버 로딩
	Car(int speed){
		this.speed = speed;
	}
	Car(String color, int speed){
		this.color = color;
		this.speed = speed;
	}
	Car(String m,String color, int speed){
		this.model = m;
		this.color = color;
		this.speed = speed;
	}
	
	// 메소드
	
	int add(int x,int y){
		int result = x + y;
		return result;
		
	}
	
	void setGas(int gas) {
		this.gas = gas;
	}
	
	boolean isLeftGas() {
		if (gas == 0) {
			System.out.println("gas가 없습니다.");
			return false;
		}
		
		System.out.println("gas가 있습니다.");
		return true;
	}
	
	void run() {
		while(true) {
			if(gas > 0) {
			System.out.println("달립니다.");
			gas -= 1;
			} 
			else {
			System.out.println("멈춥니다.");
			return ;
			}
		}

	}
}