import 'package:flutter/material.dart';
import 'package:myapp/visual_aid_screen.dart';
import 'worksheet_screen.dart';
import 'qa_screen.dart';
//import 'visual_aid_screen.dart';
import 'lesson_plan_screen.dart';
//import 'hypervocal_activity_screen.dart';
//import 'voice_to_text_screen.dart';

class FeatureScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Edu-Sahayak Features'),
        backgroundColor: Colors.deepPurple,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: GridView.count(
          crossAxisCount: 2,
          mainAxisSpacing: 16,
          crossAxisSpacing: 16,
          children: [
            _buildFeatureCard(
              context,
              title: 'Worksheet Generator',
              icon: Icons.description,
              color: Colors.orange,
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => WorksheetScreen()),
                );
              },
            ),
            _buildFeatureCard(
              context,
              title: 'Q & A',
              icon: Icons.question_answer,
              color: Colors.blue,
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => QAScreen()),
                );
              },
            ),
            _buildFeatureCard(
              context,
              title: 'Visual Aids',
              icon: Icons.image,
              color: Colors.green,
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => VisualAidScreen()),
                );
              },
            ),
            _buildFeatureCard(
              context,
              title: 'Lesson Plan',
              icon: Icons.book,
              color: Colors.purple,
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => LessonPlanScreen()),
                );
              },
            ),
            // Add more cards here if needed
          ],
        ),
      ),
    );
  }

  Widget _buildFeatureCard(
    BuildContext context, {
    required String title,
    required IconData icon,
    required Color color,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Card(
        elevation: 4,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        color: color.withOpacity(0.1),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircleAvatar(
              backgroundColor: color,
              child: Icon(icon, color: Colors.white),
            ),
            SizedBox(height: 12),
            Text(
              title,
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
