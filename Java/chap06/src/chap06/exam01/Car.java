package chap06.exam01;

public class Car {
	// �ʵ� = ����
	static	String company = "�����ڵ���";
			String model = "�׷���";
			String color = "����";
			int maxSpeed = 350;
			int porductYear;
			int speed;
			int gas;
	
	// ������
	// ���� �ε�
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
	
	// �޼ҵ�
	
	int add(int x,int y){
		int result = x + y;
		return result;
		
	}
	
	void setGas(int gas) {
		this.gas = gas;
	}
	
	boolean isLeftGas() {
		if (gas == 0) {
			System.out.println("gas�� �����ϴ�.");
			return false;
		}
		
		System.out.println("gas�� �ֽ��ϴ�.");
		return true;
	}
	
	void run() {
		while(true) {
			if(gas > 0) {
			System.out.println("�޸��ϴ�.");
			gas -= 1;
			} 
			else {
			System.out.println("����ϴ�.");
			return ;
			}
		}

	}
}