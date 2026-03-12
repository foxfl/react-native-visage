require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "Visage"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => '18.0' }
  s.source       = { :git => "https://github.com/foxfl/react-native-visage.git", :tag => "#{s.version}" }

  s.source_files = [
    "ios/**/*.{swift}",
    "ios/**/*.{m,mm}",
    "cpp/**/*.{hpp,cpp}",
  ]

  s.resources = ['ios/**/*.mlmodelc']
  s.frameworks = 'Vision', 'CoreML', 'Photos', 'Accelerate'

  s.dependency 'React-jsi'
  s.dependency 'React-callinvoker'

  load 'nitrogen/generated/ios/Visage+autolinking.rb'
  add_nitrogen_files(s)

  install_modules_dependencies(s)
end
