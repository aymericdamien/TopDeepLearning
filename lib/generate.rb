require 'yaml'
require 'time'
require_relative './utils.rb'

module Github
  def self.data(yaml:)
    read(yaml: yaml)
  end

  def self.layout(content)
    header = <<-STRING
# Top Deep Learning Projects

A list of popular github projects related to deep learning (ranked by stars).

Last Update: #{content[:last_updated].strftime("%Y.%m.%d")}

Project Name | Stars | Description |
------------ | -----:| ----------- |
    STRING

    body = content[:data]
      .sort { |l, r| r[:stars] <=> l[:stars] }
      .map { |h| "#{md_github_url(repo: h[:repo])} | #{h[:stars]} | #{h[:description]}" }
      .join("\n")

    header + body
  end

  def self.generate
    puts layout(data(yaml: 'data.yml'))
  end
end
