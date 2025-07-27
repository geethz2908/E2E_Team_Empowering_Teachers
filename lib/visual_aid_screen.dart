import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';

class VisualAidScreen extends StatefulWidget {
  @override
  _VisualAidScreenState createState() => _VisualAidScreenState();
}

class _VisualAidScreenState extends State<VisualAidScreen> {
  String? fileName;
  TextEditingController topicController = TextEditingController();
  String? selectedGrade;

  List<String> gradeOptions = [
    'Grade 1',
    'Grade 2',
    'Grade 3',
    'Grade 4',
    'Grade 5',
    'Grade 6',
    'Grade 7',
    'Grade 8',
    'Grade 9',
    'Grade 10',
    'Grade 11',
    'Grade 12',
  ];

  Future<void> pickFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf'],
    );
    if (result != null) {
      setState(() {
        fileName = result.files.single.name;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: true,
      appBar: AppBar(
        title: Text('Educational Image Generator'),
        backgroundColor: Colors.deepPurple,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20.0),
          child: Card(
            elevation: 5,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Center(
                    child: Text(
                      'Educational Image Generator',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),

                  Text(
                    'Upload PDF Textbook:',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),

                  ElevatedButton(
                    onPressed: pickFile,
                    child: Text(fileName ?? 'Choose File'),
                  ),

                  const SizedBox(height: 16),

                  Text(
                    'Topic/Subject:',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  TextField(
                    controller: topicController,
                    decoration: InputDecoration(
                      border: OutlineInputBorder(),
                      hintText: 'e.g., Newton\'s Third Law of Motion',
                    ),
                  ),

                  const SizedBox(height: 16),

                  Text(
                    'Grade Level:',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  DropdownButtonFormField<String>(
                    value: selectedGrade,
                    hint: Text('Select Grade'),
                    items:
                        gradeOptions.map((grade) {
                          return DropdownMenuItem<String>(
                            value: grade,
                            child: Text(grade),
                          );
                        }).toList(),
                    onChanged: (value) {
                      setState(() {
                        selectedGrade = value;
                      });
                    },
                    decoration: InputDecoration(border: OutlineInputBorder()),
                  ),

                  const SizedBox(height: 24),

                  ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                    onPressed: () {
                      // TODO: Add logic to generate the educational image
                    },
                    child: Text(
                      'Generate Educational Image',
                      style: TextStyle(fontSize: 16, color: Colors.white),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
