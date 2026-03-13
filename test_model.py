import unittest
from predict import TopicPredictor

class TestTopicPredictor(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Se ejecuta una vez antes de todos los tests. Carga el modelo.
        cls.predictor = TopicPredictor()

    def test_supported_payroll_query(self):
        """Prueba que un mensaje claro de nómina se clasifique correctamente"""
        # Usamos términos más directos del dominio corporativo (payroll, net pay, deductions)
        message = "Can you help me understand my payroll deductions and net pay?"
        result = self.predictor.predict(message)
        
        # Print de debug para ver qué está prediciendo exactamente
        print(f"\n[DEBUG Payroll Test] -> {result}")
        
        self.assertEqual(result["status"], "success", f"El modelo no superó el umbral. Resultado: {result}")
        self.assertEqual(result["topic"], "payroll")
        self.assertGreaterEqual(result["confidence"], 0.60)

    def test_unsupported_operation_threshold(self):
        """Prueba que un mensaje totalmente fuera de dominio sea flagueado como no soportado"""
        message = "Can you order a pepperoni pizza to the office?"
        result = self.predictor.predict(message)
        
        print(f"\n[DEBUG Unsupported Test] -> {result}")
        
        self.assertEqual(result["status"], "unsupported")
        self.assertTrue(result["confidence"] < 0.60)

    def test_single_routing(self):
        """Prueba que el modelo solo devuelve UN tópico (cumpliendo el requerimiento)"""
        message = "I need help with my taxes and benefits"
        result = self.predictor.predict(message)
        
        print(f"\n[DEBUG Single Routing Test] -> {result}")
        
        # Debe devolver un string (un solo tópico), no una lista
        self.assertIsInstance(result["topic"], str)

if __name__ == '__main__':
    unittest.main()