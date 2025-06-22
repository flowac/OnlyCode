import Foundation

func httpGet(url: String, msg: String) {
    guard let url = URL(string: url) else {
        print("\(msg): Invalid URL")
        return
    }
    
    let task = URLSession.shared.dataTask(with: url) { data, response, error in
        if let error = error {
            print("\(msg): Error: \(error)")
            return
        }
        
        if let data = data, let text = String(data: data, encoding: .utf8) {
            print("\(msg): \(text)")
        } else {
            print("\(msg): Decode Error")
        }
    }
    
    task.resume()
}

func httpPost(url: String, fileName: String, completion: @escaping (String?) -> Void) {
    guard let url = URL(string: url) else {
        print("Invalid URL")
        return
    }
    
    guard let imageData = NSData(contentsOfFile: fileName) as Data? else {
        print("Could not load file from path")
        return
    }
    
    let boundary = "Boundary-\(UUID().uuidString)"
    var body = Data()
    
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(URL(fileURLWithPath: fileName).lastPathComponent)\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
    body.append(imageData)
    body.append("\r\n".data(using: .utf8)!)
    body.append("--\(boundary)--\r\n".data(using: .utf8)!)
    
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    request.httpBody = body
    
    let task = URLSession.shared.dataTask(with: request) { data, response, error in
        if let error = error {
            print("Error: \(error)")
            return
        }
        
        if let data = data, let text = String(data: data, encoding: .utf8) {
            //print("Response: \(text)")
            completion(text)
        } else {
            print("Encode Error")
        }
    }
    
    task.resume()
}

struct JsonParser {
    private var doubles: [String: Double] = [:]
    private var currentPath: [String] = []
    
    mutating func getDoubles(from input: String) -> [String: Double] {
        doubles.removeAll()
        currentPath.removeAll()
        
        do {
            let jsonData = input.data(using: .utf8)
            let jsonObject = try JSONSerialization.jsonObject(with: jsonData!)
            traverse(jsonObject)
        } catch {
            var snip = String(input.prefix(20))
            print("JSON parsing error: \(error) - \(snip)")
            doubles.removeAll()
        }
        
        return doubles
    }
    
    private mutating func traverse(_ value: Any) {
        switch value {
        case let number as NSNumber:
            // Convert ALL numeric values (int, float, double, bool) to Double
            let path = currentPath.joined(separator: ".")
            doubles[path] = number.doubleValue
            
        case let dict as [String: Any]:
            for (key, val) in dict {
                currentPath.append(key)
                traverse(val)
                currentPath.removeLast()
            }
            
        case let array as [Any]:
            for (index, val) in array.enumerated() {
                currentPath.append("[\(index)]")
                traverse(val)
                currentPath.removeLast()
            }
            
        default:
            break
        }
    }
}
