import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:myapp/lib/utils/constants.dart';


class WorksheetPromptScreen extends StatefulWidget {
  @override
  _WorksheetPromptScreenState createState() => _WorksheetPromptScreenState();
}

class _WorksheetPromptScreenState extends State<WorksheetPromptScreen> {
  String? selectedGrade;
  String? selectedSubject;
  String? selectedChapter;
  String selectedDifficulty = 'medium'; // Default difficulty
  TextEditingController promptController = TextEditingController();
  String responseText = '';

  final List<String> grades = List.generate(12, (i) => 'Grade ${i + 1}');
  final List<String> subjects = ['Science', 'Math', 'Social', 'English'];
  final List<String> chapters = List.generate(10, (i) => 'Chapter ${i + 1}');
  final List<String> difficulties = ['easy', 'medium', 'hard'];

  Future<void> generateWorksheet() async {
    if (selectedGrade != null &&
        selectedSubject != null &&
        selectedChapter != null) {
      final url = Uri.parse(
        'http://173.212.242.240:5005/api/generate-worksheet?grade=${selectedGrade!}&subject=${selectedSubject!}&chapter=${selectedChapter!}&difficulty=$selectedDifficulty',
      );

      try {
        final response = await http.post(
          url,
          headers: <String, String>{
            'accept': 'application/json',
            'x-api-key': 'supersecretapikey',
            // Replace with actual user ID and username if available
            'x-user-id': '68820adcbc9e1f62a820077a',
            'x-username': 'admin',
          },
          body: '', // Empty body as per curl command
        );

        if (response.statusCode == 200) {
          final jsonResponse = jsonDecode(response.body);
          // Extract the nested worksheet content
          final worksheetContent = jsonResponse['worksheet'];

          // Decode the JSON string within the worksheet content
          final decodedWorksheet = jsonDecode(
            worksheetContent.substring(7, worksheetContent.length - 3),
          );

          // Format the questions nicely
          final questions = decodedWorksheet['worksheet'] as List;
          setState(() {
            responseText = questions.join('');
          });
        } else {
          setState(() {
            responseText = 'Error: ${response.statusCode} - ${response.body}';
          });
        }
      } catch (e) {
        setState(() {
          responseText = 'Error during API call: ${e.toString()}';
        });
        if (kDebugMode) {
          print('Error details: $e');
        }
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text("Please select grade, subject, and chapter."),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Widget _buildDropdown<T>({
    required String label,
    required T? selectedValue,
    required List<T> items,
    required void Function(T?) onChanged,
  }) {
    return DropdownButtonFormField<T>(
      value: selectedValue,
      decoration: InputDecoration(
        labelText: label,
        border: OutlineInputBorder(),
        isDense: true,
      ),
      items:
          items.map((item) {
            return DropdownMenuItem<T>(
              value: item,
              child: Text(item.toString()),
            );
          }).toList(),
      onChanged: onChanged,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Worksheet Prompt')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: ListView(
          children: [
            Row(
              children: [
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.only(right: 8.0),
                    child: _buildDropdown<String>(
                      label: "Grade",
                      selectedValue: selectedGrade,
                      items: grades,
                      onChanged: (val) => setState(() => selectedGrade = val),
                    ),
                  ),
                ),
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.only(left: 8.0),
                    child: _buildDropdown<String>(
                      label: "Subject",
                      selectedValue: selectedSubject,
                      items: subjects,
                      onChanged: (val) => setState(() => selectedSubject = val),
                    ),
                  ),
                ),
              ],
            ),
            SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.only(right: 8.0),
                    child: _buildDropdown<String>(
                      label: "Chapter",
                      selectedValue: selectedChapter,
                      items: chapters,
                      onChanged: (val) => setState(() => selectedChapter = val),
                    ),
                  ),
                ),
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.only(left: 8.0),
                    child: _buildDropdown<String>(
                      label: "Difficulty",
                      selectedValue: selectedDifficulty,
                      items: difficulties,
                      onChanged:
                          (val) => setState(() => selectedDifficulty = val!),
                    ),
                  ),
                ),
              ],
            ),
            SizedBox(height: 16),
            TextField(
              controller: promptController,
              maxLines: 3,
              decoration: InputDecoration(
                labelText: 'Enter your prompt/question (Optional)',
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: generateWorksheet,
              child: Text('Generate'),
            ),
            SizedBox(height: 20),
            if (responseText.isNotEmpty)
              Container(
                padding: EdgeInsets.all(12),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8),
                  color: Colors.grey[100],
                ),
                child: Text(responseText, style: TextStyle(fontSize: 16)),
              ),
          ],
        ),
      ),
    );
  }
}
