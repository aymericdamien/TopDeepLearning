require_relative './lib/updater.rb'
require_relative './lib/generate.rb'
require_relative './lib/scrapper.rb'

desc "Update stars & description"
task :update do
  Github::update
end

desc "Generate README from data.yml"
task :generate do
  Github::generate
end

desc "Status"
task :status do
  Github::status
end

desc "Ensure repo list & README has the same repos"
task :repo_diff do
  Github::repo_diff
end

desc "Scrape Readme"
task :scrape do
  Github::scrape_readme_as_yaml
end
